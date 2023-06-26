# Introduction

This repo was created for developers to use as an example of how
to use an OpenAI language model on data that it was not trained on
and that is too big to embed in the prompt.  It is not intended
to be usable other than as an example.  It uses a subset of public
IRS 990 tax returns as an example of a dataset that can be queried
using OpenaI.

These instructions assume that you are running this code in a GitHub
Codespace or devcontainer.

# Prerequisites

1. A GitHub account, which is required to use Codespaces, which are free as long as
   you stay below a certain level of utilization.
2. An OpenAI API key.
3. (Optional) An S3 bucket location containing pre-trained embeddings for the
   IRS 990 data.  This step is optional if you want to train your own
   embeddings using OpenAI.  Computing the embeddings for the returns
   for a year or two will take a couple hours and may cost up to $20 (in June, 2023).
   Obviously if you do this once, you would want to the persist the
   embeddings somewhere like S3 for re-use later.

# Instructions

These instructions assume you have created a Codespace dev container.  You can
create a Codespace container from the main page of this repository by clicking
on `Code`, then `Codespaces`, then `Create codespace on main`
(creating a codespace can take about 10 minutes).
You would type the commands described below in the Terminal of the Codespace.

1. Download or create a set of embeddings.
   * To download a set of embeddings from a public S3 bucket, type:

     ```
     get-embeddings <YOUR S3 BUCKET NAME> <YOUR BUCKET PATH>
     ```

   * If you don't have access to pre-computed embeddings, type the command below,
     which will download files for 2022 and 2023 from the IRS,
     parse the files, and call OpenAI to get the embeddings.
     It takes several hours to complete and will incur charges to
     your OpenAI account, and it may not work.  Use at your own risk.
     To do this, type:

     ```
     export OPENAI_API_KEY=<YOUR OPEN AI API KEY>
     python -m query_gpt.data
     ```

2. Load a small sample of the embeddings into Qdrant for testing (this takes less than a minute):

   ```
   load-vector-db
   ```

   The small sample is useful for development and debugging.  However, unless
   you load the full database, you won't get very meaningful query responses.
   To load the full set of the embeddings (which takes more than an hour), type:

   ```
   load-vector-db --full
   ```

   For the full dataset, I recommend using an 8G, 4-core instance.  You can
   modify the instance after it's created on the same menu where you created it.
   After the data has been loaded, you can change back to the smaller machine type.

3. Run some queries (Note that if you don't load the full set of embeddings, you won't get
   as meaningful answers):

   ```
   export OPENAI_API_KEY=<YOUR OPEN AI API KEY>
   query
   ```
   
   
   

