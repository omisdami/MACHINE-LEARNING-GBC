import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def clean_text(text):
    """Removes unnecessary whitespace and special characters from text."""
    return ' '.join(text.strip().split()) if text else "N/A"

def scrape_yelp(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to retrieve the webpage. Status Code:", response.status_code)
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract restaurant name
    name_tag = soup.find('h1', class_='y-css-olzveb')
    restaurant_name = clean_text(name_tag.text) if name_tag else "Not Found"
    
    # Extract total number of reviews
    reviews_count_tag = soup.find('span', class_='y-css-yrt0i5')
    reviews_count = clean_text(reviews_count_tag.text) if reviews_count_tag else "Not Found"
    reviews_count = re.sub(r'[^0-9]', '', reviews_count)  # Extract numerical value only
    
    # Extract reviews
    reviews = []
    review_texts = soup.find_all('p', class_='comment__09f24__D0cxf')
    reviewers = soup.find_all('a', class_='y-css-1x1e1r2')
    ratings = soup.find_all('path', attrs={"fill": "rgba(251,67,60,1)"})
    
    for i in range(len(review_texts)):
        try:
            reviewer = clean_text(reviewers[i].text) if i < len(reviewers) else "Anonymous"
            review_text = clean_text(review_texts[i].text) if i < len(review_texts) else "N/A"
            rating = "5 Stars" if i < len(ratings) else "N/A"
            
            reviews.append([reviewer, review_text, rating])
        except Exception as e:
            print(f"Error extracting review: {e}")
    
    # Save to CSV
    csv_filename = "yelp_reviews.csv"
    df = pd.DataFrame(reviews, columns=["Reviewer", "Review Text", "Rating"])
    df.insert(0, "Restaurant Name", restaurant_name)
    df.insert(1, "Total Reviews", reviews_count)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"Data successfully saved to {csv_filename}")

if __name__ == "__main__":
    url = input("Enter the Yelp restaurant URL: ")
    scrape_yelp(url)