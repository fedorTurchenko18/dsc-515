#!/bin/bash

# Script to export new environment variables
# run as:
# bash export_dotenv.bash

# Make sure shell is trusted
set -a
source .env
set +a