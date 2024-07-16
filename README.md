# Movie Recommendation system Using BERT

The Movie Recommendation System using BERT is designed to suggest movies to users based on the similarity of their metadata. This project leverages the powerful BERT model to create embeddings from movie data and uses these embeddings to find and recommend movies that are similar to a given movie.

## Tools and Technologies: Python: The primary programming language used for the project.

- Pandas and NumPy: Libraries for data manipulation and numerical computing.
- Scikit-learn: Used for preprocessing and calculating cosine similarity.
- Matplotlib and Seaborn: Tools for data visualization.
- Sentence-BERT: A pre-trained BERT model for generating sentence embeddings.
- Kaggle API: Utilized for downloading datasets.

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained NLP model developed by Google. It is designed to understand the context of a word in search queries by looking at both the words before and after it, making it bidirectional. BERT's architecture is based on transformers, which allow it to achieve state-of-the-art performance on various NLP tasks. It has been pre-trained on a vast corpus of text data and can be fine-tuned for specific tasks with minimal additional training. BERT's ability to capture context makes it particularly powerful for tasks like text classification, sentiment analysis, and question answering.

## Workflow:

Step 1: Data Collection

Data was collected from Kaggle's "The Movies Dataset," which includes comprehensive metadata on movies.

Step 2: Data Preprocessing

Metadata Cleaning: Loaded the movie metadata and filtered relevant columns such as genres, original language, production countries, tagline, original title, adult, release date, and status.

Genres and Production Countries: Processed these columns to extract and clean the data into a usable format.

Keywords and Credits: Additional datasets containing keywords and credits (cast and directors) were loaded and processed similarly.

Step 3: Data Integration

Integrated the cleaned and processed data into a single dataset. This involved merging datasets on the movie ID and handling missing values to ensure a complete and coherent dataset.

Step 4: Exploratory Data Analysis (EDA)

Performed EDA to understand the data distribution and relationships among various features:

Language Distribution: Analyzed the distribution of original languages of movies.

Release Year Distribution: Examined the number of movies released each year.

Genre Analysis: Identified the most common genres in the dataset.

Step 5: BERT Embeddings

Used the Sentence-BERT model to generate embeddings for each movie. The embeddings were based on a combination of features, including the title, genres, original language, director, keywords, cast, tagline, and production countries. This step transformed textual data into numerical representations that encapsulate the semantic meaning of the movies.

Step 6: Cosine Similarity

Calculated the cosine similarity between the embeddings to measure how similar the movies are to each other. This similarity matrix forms the core of the recommendation engine.

Step 7: Building a Movie Recommendation System

Developed a recommendation system that takes a movie as input and outputs a list of similar movies based on the cosine similarity scores. The system sorts movies by similarity and recommends the top matches.

## Conclusion:

The Movie Recommendation System using BERT successfully demonstrates how advanced NLP models can be applied to create powerful recommendation engines. By leveraging BERT embeddings and cosine similarity, the system provides accurate and relevant movie recommendations based on detailed movie metadata. This approach can be extended and refined further to include user preferences and ratings for even more personalized recommendations.








