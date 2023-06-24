# Introduction

This repo was created to serve as an example of how to use OpenAI language
model to data that it was trained when the data is too big to fit in the 
prompt.  It uses a subset of public IRS 990 tax returns as an example.

It is intended to be run in a Codespace.

# Prerequisites

1. A GitHub account (required to use Codespaces, which are free below a certain utilization)
2. An OpenAI key.
3. (Optional) An S3 bucket location containing pre-trained embeddings for the
   IRS 990 data.  This step is optional if you want to train your own
   embeddings using OpenAI.  Computing the embeddings for the returns
   for a year or two will take a couple hours and may cost up to $20 (in June, 2023).
   Obviously if you do this once, you would want to the persist the
   embeddings somewhere like S3 for re-use later.

# Instructions

These instructions assume you have started Codespace dev container.

1. Create or download a set of embeddings.
   * To down them from a public S3 bucket, type:
     `get-embeddings <bucket-name>/<path>`

   * To create your own, type:
     `<TBD>`

2. Load a small subset of the embeddings into Weaviate for testing (which takes less than a minute):
   `load-weaviate`

   Or, load the full set of the embeddings (which takes more than an hour):
   `load-weaviate --full`

   Loading the full dataset may require increasing the disk storage on the instance, and you
   may also want to increase the memory and CPUs.

3. Run some queries.
   
   
   

