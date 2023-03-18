import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from flask import Flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import tensorflow as tf

app = Flask(__name__)


def get_paragraphs(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get the recipe instructions and ingredients
    recipe = soup.find('div', class_='tasty-recipe') or soup.find('div', class_='wprm-recipe-container')
    instructions = recipe.find_all('div', class_='tasty-recipe-instructions') + recipe.find_all('div', class_='wprm-recipe-instruction-text')
    ingredients = recipe.find_all('li', class_='tasty-recipe-ingredients') + recipe.find_all('li', class_='wprm-recipe-ingredient')

    # Extract the text from the instructions and ingredients
    instructions_text = ""
    for instruction in instructions:
        instructions_text += instruction.get_text().strip() + "\n"

    ingredients_text = ""
    for ingredient in ingredients:
        ingredients_text += ingredient.get_text().strip() + "\n"

    paragraphs = [instructions_text, ingredients_text]
    paragraphs = [re.sub(r'[^\w\s.]', '', item) for item in paragraphs]
    paragraphs = [re.sub(r'[\n,\xa0.]', ' ', item) for item in paragraphs]

    return paragraphs

def find_none_paragraphs(paragraphs, fitlables):
    # create a TF-IDF vectorizer and transform the paragraphs and target paragraph
    vectorizer = TfidfVectorizer()
    fitlables_fit = vectorizer.fit_transform(paragraphs + [fitlables])

    # calculate the cosine similarity between the target paragraph and each paragraph in the list
    similarity_scores_fitlables = cosine_similarity(fitlables_fit[-1], fitlables_fit[:-1]).flatten()

    # find the index of the paragraph with the highest similarity score
    max_index_fitlables = np.argmax(similarity_scores_fitlables)
    paragraphs_none = paragraphs.copy()
    del paragraphs_none[max_index_fitlables]
    paragraph_indins = paragraphs[max_index_fitlables]
    #print(f"Most similar paragraph: {paragraphs[max_index_fitlables]}")
    # print(f"Similarity score: {similarity_scores_fitlables[max_index_fitlables]}")
    return paragraphs_none, paragraph_indins

def text_preprocess(dataset):
    df = pd.read_csv(dataset)
    # Find the index of the row with an empty "ingredients" column
    df.dropna(subset=['ingredients', 'instructions'], inplace=True)
    # Drop the row
    df['paragraphs'] = None  # new column for the paragraphs
    df['none_paragraphs'] = None
    urls_dataset = df["url"]
    result = []

    i = 0
    for url in df['url']:
        try:
            # Get the paragraphs from the URL
            paragraphs = get_paragraphs(url)
            # Find the none paragraphs
            none_paragraphs, paragraph_ind = find_none_paragraphs(paragraphs, df.loc[i, 'ingredients'])
            none_paragraphs, paragraph_ins = find_none_paragraphs(none_paragraphs, df.loc[i, 'instructions'])
            # Create a dictionary for the URL
            data = create_dictionary(url, none_paragraphs, paragraph_ind, paragraph_ins)
            # Add the dictionary to the result list

            result.append(data)
        except Exception as e:
            #print(f"Error processing URL {url}: {str(e)}")
        finally:
            i += 1
    return result
def create_dictionary(url, none_paragraphs, ingredients, instructions):
    # Create a list of paragraphs with their labels
    none_paragraphs = ''.join(none_paragraphs)
    labeled_paragraphs = [{'text': ingredients, 'label': 'ingredients'},
                          {'text': instructions, 'label': 'instructions'},
                          {'text': none_paragraphs, 'label': 'none'}]

    # Create a dictionary for the URL
    data = {
        'url': url,
        'paragraphs': labeled_paragraphs,
    }
    #print(data)
    return data
def prepare_data(recipes):
    class_labels = {'ingredients': 0, 'instructions': 1, 'none': 2}
    data = []
    for recipe in recipes:
        for paragraph in recipe['paragraphs']:
            data.append({'text': paragraph['text'], 'label': class_labels[paragraph['label']]})
    return data




def train_model(recipes, num_epochs, batch_size, hidden_size, num_classes):
    result = text_preprocess(recipes)
    data = prepare_data(result)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, [d['label'] for d in data], test_size=0.2, random_state=42)
    # Define vocabulary size
    vocab_size = 10000
    # Define embedding size
    embedding_size = 32
    # Tokenize texts
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([d['text'] for d in X_train])
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=None),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train model
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_data = X_train[i:i+batch_size]
            batch_labels = [d['label'] for d in batch_data]
            batch_text = [d['text'] for d in batch_data]
            # Tokenize batch data
            batch_sequence = tokenizer.texts_to_sequences(batch_text)
            # Pad batch sequence
            batch_sequence = tf.keras.preprocessing.sequence.pad_sequences(batch_sequence, padding='post')
            # Convert batch labels to one-hot encoding
            batch_one_hot = tf.keras.utils.to_categorical(batch_labels, num_classes)
            # Train model on batch
            model.train_on_batch(batch_sequence, batch_one_hot)
        # Evaluate model after each epoch
        loss, accuracy = model.evaluate(tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences([d['text'] for d in X_test]), padding='post'),
            tf.keras.utils.to_categorical(y_test, num_classes),
            verbose=False)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    return model



def predict(model, url):
    # Get the paragraphs from the URL
    paragraphs = get_paragraphs(url)

    # Define vocabulary size and tokenizer
    vocab_size = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(paragraphs)

    # Convert paragraphs to sequences and pad them
    recipe = tokenizer.texts_to_sequences(paragraphs)
    recipe = tf.keras.preprocessing.sequence.pad_sequences(recipe, maxlen=10000)

    # Predict
    prediction = model.predict(recipe)

    # Get the max prob and the label index for each paragraph
    max_probs = np.max(prediction, axis=1)
    label_indices = np.argmax(prediction, axis=1)

    # Extract ingredients and instructions
    ingredients = []
    instructions = []
    results = []
    for i, paragraph in enumerate(paragraphs):
        label = "Ingredient" if label_indices[i] == 0 else "Instruction" if label_indices[i] == 1 else "None"
        if label == "Ingredient":
            ingredients.append(paragraph)
        elif label == "Instruction":
            instructions.append(paragraph)
        results.append((paragraph, label))

    # Print out the results
    for result in results:
        print(f"{result[0]} - {result[1]}")

    return ingredients, instructions



def get_ingredient_weight(ingredient, quantity, unit):
    #Get the weight of the ingredient in grams from an online conversion tool
    url = "https://www.convert-me.com/en/convert/cooking/?from=cups&to=grams&a=" + ingredient.replace(' ', '+')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    result_div = soup.find('div', class_='results')
    if result_div is not None:
        weight_in_grams = float(result_div.find('span').text.strip())
        return weight_in_grams
    else:
        conversions = {
            'cups': 240,
            'tablespoon': 15,
            'teaspoon': 5,
            'quarts': 960
        }

        if quantity is not None:
            if unit in conversions:
                weight_in_grams = float(quantity) * conversions[unit]
                return weight_in_grams
            else:
                #print(f"Cannot find weight for {ingredient}")
                return None
        else:
            #print(f"Invalid quantity for {ingredient}")
            return None


def get_nutritional_info(recipe, nutrition):
    df = pd.read_csv(nutrition)
    nutrition = {}
    for item in recipe:
        #Parse the ingredient string

        parts = item.split()
        if len(parts) < 2:
            #print(f"Cannot parse ingredient {item}")
            continue

        #Skip if first word is "Ingredient"
        if parts[0] == "Ingredient":
            continue

        quantity = parts[0]
        unit = parts[1]
        ingredient = ' '.join(parts[2:])
        quantity = quantity.replace('½', '0.5')

        # Convert quantity to grams
        weight_in_grams = get_ingredient_weight(ingredient, quantity, unit)
        if weight_in_grams is None:
            weight_in_grams = 0

        # nutritional information - CSV file
        df_item = df[df['name'].str.contains('|'.join(ingredient.split()), case=False)]
        if df_item.empty:
            #Try to extract the ingredient name from the string, removing adjectives and descriptors
            ingredient_name = ingredient.split(',')[0]
            ingredient_name = ' '.join([word for word in ingredient_name.split() if word not in ['raw', 'cooked']])

            df_item = df[df['name'].str.contains(ingredient_name, case=False)]
            if df_item.empty:
                #print(f"Cannot find nutritional information for {ingredient}")
                continue

        # Calculate nutritional information based on weight
        nutrition_factors = ['calories', 'total_fat', 'saturated_fat', 'cholesterol', 'sodium',
                             'fiber', 'protein']
        for factor in nutrition_factors:
            if pd.isna(df_item[factor]).iloc[0]:
                continue
            if factor not in nutrition:
                nutrition[factor] = 0
            value_str = str(df_item[factor].iloc[0]).replace('g', '').replace('m', '')  # Remove units of measurement
            if value_str == '':
                continue
            nutrition[factor] += weight_in_grams / 100 * float(value_str)
    return nutrition

import re

def parse_ingredients(ingredients_str):
    parsed_ingredients = []
    for ingredient in ingredients_str:
        ingredient = ingredient.strip()
        quantity = re.findall(r'\d+', ingredient)
        quantity = int(quantity[0]) if quantity else None
        name = re.findall(r'[a-zA-Z]+', ingredient)
        name = ' '.join(name).strip() if name else None
        if quantity and name:
            parsed_ingredients.append((name, quantity))
    return parsed_ingredients

if __name__ == "__main__":
    model = train_model('loaveandlemons_dataset.csv', num_epochs=10, batch_size=32, hidden_size=100, num_classes=3)

    # URLs to extract paragraphs from
    urls = [
        'https://www.loveandlemons.com/homemade-pasta-recipe/'
    ]
    for url in urls:
        ingredients, instructions = predict(model, url)
        parse_ingredients(ingredients)
        #ingredients = ['2 cups all-purpose flour, spooned & leveled','3 large eggs','0.5 teaspoon sea salt']

        nutrition = get_nutritional_info(ingredients, 'nutrition.csv')

        # Create the final JSON object
        result = {
            "Recipe": ingredients,
            "Nutritional information": nutrition,
            "INSTRUCTIONS": "\n".join(instructions)
        }
        json_results = json.dumps(result, indent=4)
        print(json_results)
