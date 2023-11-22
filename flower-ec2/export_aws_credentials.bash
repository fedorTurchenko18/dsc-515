#!/bin/bash
# Run `chmod +x export_aws_credentials.sh` to make this script executable

# Run export_dotenv.sh from the directory one level above
# to ensure that the latest environment variables are exported
PARENT_DIR="$(dirname "$(pwd)")"
echo "$PARENT_DIR"
EXPORT_DOTENV_SCRIPT="$PARENT_DIR/export_dotenv.bash"

# Check if export_dotenv.bash exists and is executable
if [ -f "$EXPORT_DOTENV_SCRIPT" ] && [ -x "$EXPORT_DOTENV_SCRIPT" ]; then
    # Execute export_dotenv.bash
    cd "$PARENT_DIR"
    source "$EXPORT_DOTENV_SCRIPT"
    echo "Environment variables from export_dotenv.sh were successfully exported."
else
    echo "Error: export_dotenv.bash script not found or not executable."
    exit 1
fi


# Set the path for the credentials file
CREDENTIALS_PATH="$HOME/.aws"
CREDENTIALS_FILE="credentials"

# Check if the credentials path already exists, and create it if not
if [ ! -f "$CREDENTIALS_PATH" ]; then
    mkdir "$CREDENTIALS_PATH"
fi

# Check if the credentials file already exists, and create it if not
if [ ! -f "$CREDENTIALS_PATH/$CREDENTIALS_FILE" ]; then
    touch "$CREDENTIALS_PATH/$CREDENTIALS_FILE"
fi

# Add AWS credentials to the credentials file
echo "[default]" > "$CREDENTIALS_PATH/$CREDENTIALS_FILE"
echo "aws_access_key_id=$AWS_LAB_ACCESS_KEY" >> "$CREDENTIALS_PATH/$CREDENTIALS_FILE"
echo "aws_secret_access_key=$AWS_LAB_SECRET_ACCESS_KEY" >> "$CREDENTIALS_PATH/$CREDENTIALS_FILE"
echo "aws_session_token=$AWS_LAB_SESSION_TOKEN" >> "$CREDENTIALS_PATH/$CREDENTIALS_FILE"

echo "AWS credentials added successfully to $CREDENTIALS_PATH/$CREDENTIALS_FILE"


# Set the path for the config file
CONFIG_PATH="$HOME/.aws"
CONFIG_FILE="config"

# Check if the config path already exists, and create it if not
if [ ! -f "$CONFIG_PATH" ]; then
    mkdir "$CONFIG_PATH"
fi

# Check if the config file already exists, and create it if not
if [ ! -f "$CONFIG_PATH/$CONFIG_FILE" ]; then
    touch "$CONFIG_PATH/$CONFIG_FILE"
fi

echo "[default]" > "$CONFIG_PATH/$CONFIG_FILE"
echo "region=$AWS_REGION" >> "$CONFIG_PATH/$CONFIG_FILE"

echo "AWS region added successfully to $CONFIG_PATH/$CONFIG_FILE"
PEM_KEY_PATH="aws_management"
# Having AWS configuration being set-up, generate .pem file (only if does not exist)
if [ ! -f "$PEM_KEY_PATH/$CREDENTIALS_FILE" ]; then
    cd "$(pwd)"
    aws configure list
    aws ec2 create-key-pair --key-name FlowerKey --query 'KeyMaterial' --output text > ./flower-ec2/aws_management/$AWS_KEY_PAIR.pem
    chmod 400 ./flower-ec2/aws_management/$AWS_KEY_PAIR.pem
fi