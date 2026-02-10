import os
import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------
# Data download (safe for script execution)
# ---------------------------------------------------
ZIP_URL = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
ZIP_FILE = "book-crossings.zip"

if not os.path.exists("BX-Books.csv"):
    if not os.path.exists(ZIP_FILE):
        import urllib.request
        urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall()

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
df_books = pd.read_csv(
    "BX-Books.csv",
    sep=";",
    encoding="ISO-8859-1",
    usecols=["ISBN", "Book-Title", "Book-Author"],
    dtype=str
).rename(columns={
    "ISBN": "isbn",
    "Book-Title": "title",
    "Book-Author": "author"
})

df_ratings = pd.read_csv(
    "BX-Book-Ratings.csv",
    sep=";",
    encoding="ISO-8859-1",
    usecols=["User-ID", "ISBN", "Book-Rating"],
    dtype={"User-ID": np.int32, "ISBN": str, "Book-Rating": np.float32}
).rename(columns={
    "User-ID": "user",
    "ISBN": "isbn",
    "Book-Rating": "rating"
})

# ---------------------------------------------------
# Filter sparse noise (this matters a LOT)
# ---------------------------------------------------
user_counts = df_ratings["user"].value_counts()
book_counts = df_ratings["isbn"].value_counts()

df_ratings = df_ratings[
    df_ratings["user"].isin(user_counts[user_counts >= 200].index) &
    df_ratings["isbn"].isin(book_counts[book_counts >= 100].index)
]

# ---------------------------------------------------
# Merge & pivot
# ---------------------------------------------------
df = df_ratings.merge(df_books, on="isbn")

book_user_matrix = df.pivot_table(
    index="title",
    columns="user",
    values="rating"
).fillna(0)

# ---------------------------------------------------
# Sparse matrix + index mapping
# ---------------------------------------------------
matrix = csr_matrix(book_user_matrix.values)

title_to_index = {title: i for i, title in enumerate(book_user_matrix.index)}
index_to_title = {i: title for title, i in title_to_index.items()}

# ---------------------------------------------------
# KNN model
# ---------------------------------------------------
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6  # book itself + 5 recommendations
)
model.fit(matrix)

# ---------------------------------------------------
# Recommendation function (FCC tested)
# ---------------------------------------------------
def get_recommends(book=""):
    if book not in title_to_index:
        return [book, []]

    idx = title_to_index[book]
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=6)

    recommendations = []
    for dist, i in zip(distances[0][1:], indices[0][1:]):
        recommendations.append([index_to_title[i], float(dist)])

    return [book, recommendations]


# ---------------------------------------------------
# Test (provided by FCC)
# ---------------------------------------------------
if __name__ == "__main__":
    books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    print(books)

    def test_book_recommendation():
        test_pass = True
        recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")

        if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
            test_pass = False

        recommended_books = [
            "I'll Be Seeing You",
            "The Weight of Water",
            "The Surgeon",
            "I Know This Much Is True"
        ]
        recommended_books_dist = [0.8, 0.77, 0.77, 0.77]

        for i in range(2):
            if recommends[1][i][0] not in recommended_books:
                test_pass = False
            if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
                test_pass = False

        if test_pass:
            print("You passed the challenge! 🎉🎉🎉🎉🎉")
        else:
            print("You haven't passed yet. Keep trying!")

    test_book_recommendation()
