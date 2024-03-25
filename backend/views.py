from django.shortcuts import render
from django.http import JsonResponse
from sklearn.metrics import accuracy_score
import os
from django.conf import settings
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
import nltk
from scipy.sparse import hstack
from nltk import pos_tag, word_tokenize

# Download specific NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def home(request):
    return render(request, 'index.html')

def main(request):
    return render(request, 'main.html')


# Define QUESTION_MAPPING if not defined
QUESTION_MAPPING = {
    'ques1': 'Describe a moment when you felt overwhelming happiness or joy.',
    'ques2': 'Share your personal opinion on the impact of social media on society.',
    'ques3': 'Discuss a cultural tradition that is significant to you.',
    'ques4': 'Share your thoughts on a recent local news event.',
    'ques5': 'Describe a memorable travel experience and its impact on your life.',
    'ques6': 'Discuss the role of technology in modern education.',
    'ques7': 'Tell me about yourself.'
}

def predict_label(request):
    if request.method == 'POST':
        # Construct the path to the CSV file within the static directory
        csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'Response.csv')
        try:
            # Read and preprocess the CSV data
            Corpus = pd.read_csv(csv_file_path, encoding='latin-1')
            Corpus['Questions'] = Corpus['Questions'].str.lower()
            Corpus['Answers'] = Corpus['Answers'].str.lower()
            Corpus['Questions'] = Corpus['Questions'].apply(word_tokenize)
            Corpus['Answers'] = Corpus['Answers'].apply(word_tokenize)
            tag_map = defaultdict(lambda: wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV

            word_Lemmatized = WordNetLemmatizer()

            Corpus['Questions_final'] = Corpus['Questions'].apply(lambda entry: ' '.join(
                [word_Lemmatized.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(entry) if word not in stopwords.words('english') and word.isalpha()]
            ))

            Corpus['Answers_final'] = Corpus['Answers'].apply(lambda entry: ' '.join(
                [word_Lemmatized.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(entry) if word not in stopwords.words('english') and word.isalpha()]
            ))

            # Train and evaluate the model
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus[['Questions_final', 'Answers_final']],
                                                                                Corpus['Generated'], test_size=0.3, random_state=19)

            Tfidf_vect_questions = TfidfVectorizer(max_features=5000)
            Tfidf_vect_answers = TfidfVectorizer(max_features=5000)

            Train_X_questions_Tfidf = Tfidf_vect_questions.fit_transform(Train_X['Questions_final'])
            Train_X_answers_Tfidf = Tfidf_vect_answers.fit_transform(Train_X['Answers_final'])
            Test_X_questions_Tfidf = Tfidf_vect_questions.transform(Test_X['Questions_final'])
            Test_X_answers_Tfidf = Tfidf_vect_answers.transform(Test_X['Answers_final'])

            Train_X_Tfidf = hstack((Train_X_questions_Tfidf, Train_X_answers_Tfidf))
            Test_X_Tfidf = hstack((Test_X_questions_Tfidf, Test_X_answers_Tfidf))

            Encoder = LabelEncoder()
            Train_Y = Encoder.fit_transform(Train_Y)
            Test_Y = Encoder.transform(Test_Y)

            SVM = svm.SVC(C=1.0, kernel='linear')
            SVM.fit(Train_X_Tfidf, Train_Y)
            predictions_SVM = SVM.predict(Test_X_Tfidf)
            accuracy = accuracy_score(predictions_SVM, Test_Y) * 100
            print("SVM Accuracy Score:", accuracy)

        except FileNotFoundError:
            return JsonResponse({'error': 'CSV file not found'}, status=404)

        # Process the form data
        question_code = request.POST.get('question')  # Get the selected question code
        answer = request.POST.get('answer')  # Get the answer

        # Map the question code to its description
        question = QUESTION_MAPPING.get(question_code, 'Unknown question')

        # Preprocess the new question and answer
        text_question = question.lower()
        text_answer = answer.lower()
        question_tokens = word_tokenize(text_question)
        answer_tokens = word_tokenize(text_answer)
        final_question = ' '.join(
            [word_Lemmatized.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(question_tokens) if word not in stopwords.words('english') and word.isalpha()]
        )
        final_answer = ' '.join(
            [word_Lemmatized.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(answer_tokens) if word not in stopwords.words('english') and word.isalpha()]
        )

        # Transform the new question and answer using TF-IDF vectorization
        new_text_question_tfidf = Tfidf_vect_questions.transform([final_question])
        new_text_answer_tfidf = Tfidf_vect_answers.transform([final_answer])
        new_text_tfidf = hstack((new_text_question_tfidf, new_text_answer_tfidf))

        # Predict the label for the new data using the trained SVM model
        prediction = SVM.predict(new_text_tfidf)
        predicted_label = "human" if prediction[0] == 1 else "AI"

        # Return the prediction and accuracy
        response_data = {
            'prediction': predicted_label,
            'accuracy': accuracy
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
