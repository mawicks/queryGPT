from glob import glob
import logging
import os
import requests
import tempfile
import xml.etree.ElementTree as ET

from tqdm import tqdm
from query_gpt.config import DATA_DIR

from query_gpt.embeddings import compute_search_embeddings

logger = logging.getLogger(__name__)

IRS_FILE_SEGMENTS = {
    2022: [
        "01A",
        "01B",
        "01C",
        "01D",
        "01E",
        "01F",
        "11A",
        "11B",
        "11C",
    ],
    2023: [
        "01A",
        "02A",
        "03A",
        "04A",
        "05A",
        "05B",
    ],
}
IRS_FILE_TEMPLATE = (
    "https://apps.irs.gov/pub/epostcard/990/xml/{year}/{year}_TEOS_XML_{segment}.zip"
)
FILENAME_GLOB = "2023*_public.xml"

NS = {"irs": "http://www.irs.gov/efile"}
RETURN_TYPES_TO_SKIP = ("990PF", "990T")


def download_and_parse_segment(year, segment):
    docs = []
    url = IRS_FILE_TEMPLATE.format(year=year, segment=segment)
    request_result = requests.get(url)

    with tempfile.NamedTemporaryFile("wb") as zip_file:
        zip_file.write(request_result.content)

        with tempfile.TemporaryDirectory() as extract_dir:
            # Python zipfile cannot handle several IRS files so use command line (:)
            os.system(f"unzip -d {extract_dir} {zip_file.name} > /dev/null ")
            for filename in glob(os.path.join(extract_dir, "*.xml")):
                doc = parse(filename)
                if doc is not None:
                    docs.append(doc)

    return docs


def combine(items):
    combined = ""
    for field in items:
        if field is not None:
            if combined:
                combined += "; "
            combined += field.text
    return combined


def parse(filename):
    root = ET.parse(filename).getroot()

    # Skip certain types of returns
    return_type = root.find(".//irs:ReturnTypeCd", NS)
    if return_type is not None and return_type.text in RETURN_TYPES_TO_SKIP:
        return

    # Skip non US addresses
    foreign_address = root.find(".//irs:ForeignAddress", NS)
    if foreign_address is not None:
        return

    doc = {}
    ein = root.find(".//irs:Filer/irs:EIN", NS)
    doc["EIN"] = ein.text

    tax_year = root.find(".//irs:TaxYr", NS)
    doc["Tax Year"] = tax_year.text

    tax_period_start = root.find(".//irs:TaxPeriodBeginDt", NS)
    tax_period_end = root.find(".//irs:TaxPeriodEndDt", NS)
    doc["Tax Period"] = f"{tax_period_start.text} to {tax_period_end.text}"

    name1 = root.find(".//irs:BusinessName/irs:BusinessNameLine1Txt", NS)
    name2 = root.find(".//irs:BusinessName/irs:BusinessNameLine2Txt", NS)
    doc["Name"] = f"{name1.text}{' ' + name2.text if name2 is not None else '' }"

    address = root.find(".//irs:USAddress/irs:AddressLine1Txt", NS).text
    city = root.find(".//irs:USAddress/irs:CityNm", NS).text
    state = root.find(".//irs:USAddress/irs:StateAbbreviationCd", NS).text
    zip = root.find(".//irs:USAddress/irs:ZIPCd", NS).text
    doc["Address"] = f"{address}, {city}, {state} {zip}"

    mission = root.find(".//irs:MissionDesc", NS)
    desc = root.find(".//irs:IRS990/irs:Desc", NS)
    primary_purpose = root.find(".//irs:PrimaryExemptPurposeTxt", NS)
    doc["Purpose"] = combine((mission, desc, primary_purpose))

    activity_paths = [
        "irs:ActivityOrMissionDesc",
        "irs:SummaryOfDirectChrtblActyGrp/irs:Description1Txt",
        "irs:SummaryOfDirectChrtblActyGrp/irs:Description2Txt",
        "irs:ProgSrvcAccomActy2Grp/irs:Desc",
        "irs:ProgSrvcAccomActy3Grp/irs:Desc",
    ]
    activities = [root.find(f"..//{path}", NS) for path in activity_paths]
    combined_activities = combine(activities)

    if combined_activities:
        doc["Activities"] = combined_activities

    website = root.find(".//irs:WebsiteAddressTxt", NS)
    doc["Website"] = website.text if website is not None else "None"

    program_accomplishments = root.findall(".//irs:DescriptionProgramSrvcAccomTxt", NS)
    if program_accomplishments:
        doc["Accomplishments"] = combine(program_accomplishments)

    program_service_revenue = root.findall(
        ".//irs:ProgramServiceRevenueGrp/irs:Desc", NS
    )
    if program_service_revenue:
        doc["Revenue"] = combine(program_service_revenue)

    expenses = root.findall(".//irs:OtherExpensesGrp/irs:Desc", NS)
    if expenses:
        doc["Expenses"] = combine(expenses)

    return doc


ORDERED_FIELDS = [
    "EIN",
    "Tax Year",
    "Tax Period",
    "Name",
    "Address",
    "Purpose",
    "Activities",
    "Website",
    "Accomplishments",
    "Revenue",
    "Expenses",
]


# Even though the dictionary is ordered, we persist these documents as JSON in the vector
# database which destroys order.  If we care about the order, we need to specify it explicitly.
def doc_to_string(doc):
    return "".join(
        [f"{key}: {doc[key]}\n" for key in ORDERED_FIELDS if doc.get(key) is not None]
    )


if __name__ == "__main__":
    for year in (2022, 2023):
        logger.info(f"Processing year: {year}")
        for segment in tqdm(IRS_FILE_SEGMENTS[year]):
            logger.info(f"Downloading segment: {segment} ({year})")
            docs = download_and_parse_segment(year, segment)
            logger.info(f"Parsed {len(docs):,d} documents in segment {segment}")

            compute_search_embeddings(
                docs,
                doc_to_string,
                data_dir=DATA_DIR,
                year=year,
                segment=segment,
            )
            logger.info("done")
