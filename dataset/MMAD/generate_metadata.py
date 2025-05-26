import json
import csv

# Read JSON file
with open('mmad.json', 'r') as f:
    data = json.load(f)

# Open CSV file for writing
with open('metadata.csv', 'w', newline='') as csvfile:
    fieldnames = ['query_image', 'question', 'options', 'answer', 'template_image', 'mask']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Iterate through JSON data
    for key, value in data.items():
        query_image = key
        mask = value.get('mask_path', '')
        if mask:
            # Add the prefix of query_image to mask_path
            mask = '/'.join(query_image.split('/')[:3]) + '/' + mask
        template_image = value.get('similar_templates', [''])[0]  # Get the first item from similar_templates

        for conversation in value.get('conversation', []):
            question = conversation.get('Question', '')
            options = '\n'.join([f"{k}: {v}" for k, v in conversation.get('Options', {}).items()])
            answer = conversation.get('Answer', '')

            writer.writerow({
                'query_image': query_image,
                'question': question,
                'options': options,
                'answer': answer,
                'mask': mask,
                'template_image': template_image
            })

print("CSV file has been generated")
