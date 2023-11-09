#Lambda Function 1

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']## TODO: fill in
    bucket = event['s3_bucket'] ## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
  
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# Lambda function 2

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-11-08-08-47-02-224"## TODO: fill in

def lambda_handler(event, context):
    
    event_body = event["body"]

    # Decode the image data
    image = base64.b64decode(event_body["image_data"])

    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)
    # For this model, the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    # Make a prediction
    inferences = predictor.predict(image)
    # Parse the inferences as a JSON object
    inferences_json = json.loads(inferences.decode("utf-8"))

    # Construct the response as a dictionary
    response_data = {
        "image_data": event_body["image_data"],
        "s3_bucket": event_body["s3_bucket"],
        "s3_key": event_body["s3_key"],
        "inferences": inferences_json,  # Use the parsed JSON object here
    }
    # Return the response as a JSON object in the body
    return {
        "statusCode": 200,
        "body": response_data  # Return the entire response data as a JSON object
    }


#Lambda function 3
import json

THRESHOLD = 0.80

def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event['body']["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(list(inferences)) > THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ("THRESHOLD_CONFIDENCE_NOT_MET")

    return {"statusCode": 200, "body": json.dumps(event)}
