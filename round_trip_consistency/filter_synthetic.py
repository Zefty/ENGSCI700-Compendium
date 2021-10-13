##############################################################################################################################
# Load libraries 
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from scipy import spatial
import torch
import csv
##############################################################################################################################
# Set up round-trip consistency method leveraging distilbert 
# QA model 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Sentence transformer model
modelSentence = SentenceTransformer('bert-base-nli-mean-tokens')

# Create function for generating answer from question and context 
def GenerateAnswer(question, context):
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt", max_length = 512, truncation = True)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer
##############################################################################################################################
# Setup output data
date = '2021-03-08'
synthetic_QA_round_trip = open('../data/' + date + '/synthetic_QA_round_trip.csv', 'wt', newline ='')
synthetic_QA_round_trip_writer = csv.writer(synthetic_QA_round_trip, delimiter = ',')
synthetic_QA_round_trip_writer.writerow(['passage', 'question', 'answer'])

# Read synthetic QA data and do round-trip 
with open('../data/' + date + '/synthetic_QA.csv', 'r') as f_in:
    reader = csv.DictReader(f_in)

    # Read each synthetic QA pair 
    for row in reader:

        # Generate answer from question and context using distillbert 
        answer = None 
        try: 
            answer = GenerateAnswer(row['question'], row['context'])
        except Exception as e: 
            print(e)
        # print(answer)

        # Convert synthetic answer and distillbert answer into dense embedding using sentence transformer 
        # And calculate cosine similarity 
        sentence_embeddings = modelSentence.encode([answer, row['answer']])
        cos_sim = 1 - spatial.distance.cosine(sentence_embeddings[0], sentence_embeddings[1])
        # print(cos_sim)

        # If similarity of answers are above a threshold we accept generated synthetic QA pair 
        # Need to optimise threshold 
        if cos_sim > 0.8:
            synthetic_QA_round_trip_writer.writerow([row['context'], row['question'], row['answer']]) 








