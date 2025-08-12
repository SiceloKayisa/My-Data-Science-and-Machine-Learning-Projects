## Book Recommendation Engine using KNN

This project was completed as part of my coursework from **FreeCodeCamp: Machine Learning with Python**

You will be working on this project with [Google Colab](https://colab.research.google.com/drive/1Neq9kkKsk-mYGdH8gwclFsjnqzA7pZD_#scrollTo=jd2SLCh8oxMh)

After going to that link, create a copy of the notebook either in your own account or locally. Once you complete the project and it passes the test (included at that link), submit your project link below. If you are submitting a Google Colaboratory link, make sure to turn on link sharing for "anyone with the link."

We are still developing the interactive instructional content for the machine learning curriculum. For now, you can go through the video challenges in this certification. You may also have to seek out additional learning resources, similar to what you would do when working on a real-world project.

**In this challenge, you will create a book recommendation algorithm using K-Nearest Neighbors.**

In this project, you will use the Book-Crossings dataset, which contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users. The dataset is already imported in the notebook, so no additional download is required.

Use NearestNeighbors from sklearn.neighbors to develop a model that shows books that are similar to a given book. The Nearest Neighbors algorithm measures the distance to determine the “closeness” of instances.

Create a function named get_recommends that takes a book title (from the dataset) as an argument and returns a list of 5 similar books with their distances from the book argument
This code:

```get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")```
should return:

```
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

Notice that the data returned from get_recommends() is a list. The first element in the list is the book title passed into the function. The second element in the list is a list of five more lists. Each of the five lists contains a recommended book and the distance from the recommended book to the book passed into the function.

If you graph the dataset (optional), you will notice that most books are not rated frequently. To ensure statistical significance, remove from the dataset users with less than 200 ratings and books with less than 100 ratings.

The first three cells import libraries you may need and the data to use. The final cell is for testing. Write all your code in between those cells.

### Project Achievements 

- **Data Acquisition and Preparation:** Successfully read and parsed raw data from CSV files (ratings.csv and books.csv), handled specific file attributes like encoding (ISO-8859-1) and column separators (;).

- **Data Cleaning and Filtering:** Performed crucial data cleaning steps by filtering out infrequent users (those with fewer than 200 ratings) and unpopular books (those with fewer than 100 ratings). This is an essential step in building an effective recommendation system.

- **Data Transformation:** Transformed the raw ratings data into a user-item matrix using pandas.pivot(). This is a foundational step for collaborative filtering, where rows represent items (books), columns represent users, and values represent ratings.

- **Machine Learning Model Implementation:** Successfully implemented a collaborative filtering model using NearestNeighbors from scikit-learn. This demonstrates the ability to choose an appropriate machine learning algorithm for a specific task.

- **Practical Application:** You created a functional recommendation engine (get_recommends) that can take a book title as input and return a list of recommended books based on cosine similarity, which is a powerful metric for finding similar items.

- **Code Testing and Validation:** Implemented a dedicated test function (test_book_recommendation) to automatically verify the correctness of your recommendation system. This demonstrates a good practice in software development by ensuring the model's output is consistent and accurate.
