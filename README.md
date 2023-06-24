# Introduction

This repo was created to serve as an example of how to use OpenAI language
model to data that it was trained when the data is too big to fit in the 
prompt.  It uses a subset of public IRS 990 tax returns as an example.

It is intended to be run in a Codespace.

# Prerequisites

1. You need an OpenAI key.
2. (Optional) An S3 bucket location containing pre-trained embeddings for the
   IRS 990 data.  This step is optional if you want to train your own
   embeddings using OpenAI.  Computing the embeddings for the returns
   for a year or two will take a couple hours and may cost up to $20 (in June, 2023).
   Obviously if you do this once, you would want to the persist the
   embeddings somewhere like S3 for re-use later.

# Instructions

These instructions apply in a Codespace dev container.

1. TBD...
