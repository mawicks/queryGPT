from collections import Counter
from glob import glob
import logging
import os
import numpy as np
import requests
import tempfile

import tiktoken
import xml.etree.ElementTree as ET


from tqdm import tqdm
from query_gpt.config import MODEL, INPUT_TOKEN_GOAL
from query_gpt.config import DATA_DIR

from query_gpt.embeddings import compute_search_embeddings


logger = logging.getLogger(__name__)
counters = Counter()

WANT_EMBEDDINGS = False

# Encoder to use to pre-check the token count when building the prompt
ENCODER = tiktoken.encoding_for_model(MODEL)

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
RETURN_TYPES_TO_SKIP = ("990PF", "990T", "990N")

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
            combined += field
    return combined

def add_namespaces_to_path(path: str):
    return ".//" + '/'.join(map(lambda s: 'irs:'+s, path.split('/')))

# TODO fix type hints to use a type variable for field_type.
def get_field(root: ET.Element, path:str, field_type: type  = str) -> object | None:
    element = root.find(add_namespaces_to_path(path), NS)
    if element is None:
        return None
    try:
        field_value = field_type(element.text)
    except Exception:
        field_value = element.text
    return field_value

def get_all_fields(root: ET.Element, path:str):
    elements = root.findall(add_namespaces_to_path(path), NS)
    if elements is None:
        return None
    return [element.text for element in elements]

def parse(filename):
    root = ET.parse(filename).getroot()

    # Skip certain types of returns
    return_type = get_field(root, 'ReturnTypeCd')
    if return_type is not None and return_type in RETURN_TYPES_TO_SKIP:
        return

    counters[return_type] += 1

    # Skip non US addresses
    foreign_address = get_field(root, 'ForeignAddress')
    if foreign_address is not None:
        return

    doc = {}
    doc["Return Type"] = return_type

    doc["EIN"] = get_field(root, "Filer/EIN")

    doc["Tax Year"] = get_field(root, 'TaxYr')


    tax_period_start = get_field(root, "TaxPeriodBeginDt")
    tax_period_end = get_field(root, "TaxPeriodEndDt")
    doc["Tax Period"] = f"{tax_period_start} to {tax_period_end}"

    name1 = get_field(root, 'BusinessName/BusinessNameLine1Txt')
    name2 = get_field(root, 'BusinessName/BusinessNameLine2Txt')
    doc["Name"] = f"{name1}{' ' + str(name2) if name2 is not None else '' }"

    address = get_field(root, "USAddress/AddressLine1Txt")
    city = get_field(root, "USAddress/CityNm")
    state = get_field(root, "USAddress/StateAbbreviationCd")
    zip = get_field(root, "USAddress/ZIPCd")
    doc["Address"] = f"{address}, {city}, {state} {zip}"

    mission = get_field(root, "MissionDesc")
    desc = get_field(root, "IRS990/Desc")
    primary_purpose = get_field(root, "PrimaryExemptPurposeTxt")
    doc["Purpose"] = combine((mission, desc, primary_purpose))

    activity_paths = [
        "ActivityOrMissionDesc",
        "SummaryOfDirectChrtblActyGrp/Description1Txt",
        "SummaryOfDirectChrtblActyGrp/Description2Txt",
        "ProgSrvcAccomActy2Grp/Desc",
        "ProgSrvcAccomActy3Grp/Desc",
    ]
    activities = [get_field(root, path) for path in activity_paths]
    combined_activities = combine(activities)

    if combined_activities:
        doc["Activities"] = combined_activities

    website = get_field(root, "WebsiteAddressTxt")
    doc["Website"] = website if website is not None else "None Provided"

    program_accomplishments = get_all_fields(root, "DescriptionProgramSrvcAccomTxt")
    if program_accomplishments:
        doc["Accomplishments"] = combine(program_accomplishments)

    program_service_revenue = get_all_fields(root, "ProgramServiceRevenueGrp/Desc")
    if program_service_revenue:
        doc["Revenue Categories"] = combine(program_service_revenue)

    expenses = get_all_fields(root, "OtherExpensesGrp/Desc")
    if expenses:
        doc["Expense Categories"] = combine(expenses)

    total_revenue = get_field(root, "TotalRevenueAmt", float)
    if total_revenue is None:
        total_revenue = get_field(root,"CYTotalRevenueAmt", float)
    doc["Total Revenue"] = total_revenue

    total_expenses = get_field(root, "TotalExpensesAmt", float)
    if total_expenses is None:
        total_expenses = get_field(root, "CYTotalExpensesAmt",float)
    doc["Total Expenses"] = total_expenses

    doc["Employee Count"] = get_field(root, "TotalEmployeeCnt",int)
    doc["Volunteer Count"] = get_field(root, "TotalVolunteersCnt", int)


    return doc


ORDERED_FIELDS = [
    "Return Type",
    "EIN",
    "Tax Year",
    "Tax Period",
    "Name",
    "Address",
    "Purpose",
    "Activities",
    "Website",
    "Accomplishments",
    "Revenue Categories",
    "Expense Categories",
    "Total Revenue",
    "Total Expenses",
    "Employee Count",
    "Volunteer Count",
]


# Even though the dictionary is ordered, we persist these documents as JSON in the vector
# database which destroys order.  If we care about the order, we need to specify it explicitly.
def doc_to_string(doc):
    return "".join(
        [f"{key}: {doc[key]}\n" for key in ORDERED_FIELDS if doc.get(key) is not None]
    )



def make_prompt(question: str, items: list[dict[str, str]], failures: int) -> str:
    """
    Given a question about the survey data, design a prompt for
    openai that should produce an answer to the question.
    The context is the additional information to provide.
    Arguments:
        question: str - Question about the survey data
        items: list[dict[str,str]] - Documents to be queried
        failures: int - Number of API calls that have failed because of reponse length.
    """

    prefix = (
        "The following records contain information taken from tax records "
        "for non-profit organizations operating in the US. "
        "The data for a single organization is delimited by <record> and </record>. "
        "At the end of these records, there is a question for you to answer about "
        "these non-profit organizations."
        "Try to keep the total response below 500 words.\n"
    )

    instruction = (
        "Please answer the question below about non-profit organizations.  "
        "Some of the records above may not be relevant to the question.  Pleas ignore "
        "any irrelevant records.  The most relevant ones may be near the top of the list. "
        "Remember to keep "
        "the response below 500 words. If you are asked to provide a list, you may "
        "need to omit some items from the list.  "
        "If so, state the the list is represntative and not complete. "
        "Capitalize any responses appropriately, even if the source data was presented in ALL CAPS. "
        f"{'Be *EXTREMELY* BRIEF in your answer. ' if failures > 0 else ''}"
        "Answer the question precisely and exclude any records that are not relevant to the question. "
        "The answer should be responsive. "
        " It's better to provide no response than to provide a response with irrelevant information. "
        "Base your answer primarily on the records above, but you may fill in "
        "holes based on any prior knowledge you have of these organizations.\n"
        f"Question: {question}\n"
        "Answer: "
    )
    formatted_items = [f"<record>\n{doc_to_string(item)}</record>\n" for item in items]

    # When counting fixed strings, two for the separators we'll add later.
    fixed_count = len(ENCODER.encode(prefix)) + len(ENCODER.encode(instruction)) + 2

    # When counting item tokens, add one for the separator we'll add later.
    variable_counts = [len(ENCODER.encode(item)) + 1 for item in formatted_items]
    allowed_item_count = sum(
        np.cumsum(variable_counts) < (INPUT_TOKEN_GOAL / (2**failures) - fixed_count)
    )

    context = "\n".join(formatted_items[:allowed_item_count])
    prompt = f"{prefix}\n{context}\n{instruction}"
    token_count = len(ENCODER.encode(prompt))
    logger.info(f"tiktoken token estimate: {token_count}")
    return prompt


if __name__ == "__main__":
    for year in (2022, 2023):
        logger.info(f"Processing year: {year}")
        for segment in tqdm(IRS_FILE_SEGMENTS[year]):
            logger.info(f"Downloading segment: {segment} ({year})")
            docs = download_and_parse_segment(year, segment)
            logger.info(f"Parsed {len(docs):,d} documents in segment {segment}")

            if WANT_EMBEDDINGS:
                compute_search_embeddings(
                    docs,
                    doc_to_string,
                    data_dir=os.path.join(DATA_DIR, "embeddings"),
                    year=year,
                    segment=segment,
                )
            logger.info(f"{counters}")
            logger.info("done")

