import os
import random
import re
import sys
from typing import Counter
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create a dict for the model
    model = {}
    page = page[0]
    # If the page has links
    if len(corpus[page]) != 0:
        # The probability of clicking that link is equal to the d factor by the amount of links the pages has, plus 1-d by the amount of total pages
        link_prob = (damping_factor / len(corpus[page])) + ((1 - damping_factor) / len(corpus))
        # For every link in the corpus
        for key in corpus:
            # If that link is in the list of links of the page
            if key in corpus[page]:
                # The probability is equal to the previous calculated
                model[key] = link_prob
            # If that link is not in the list of links of the page
            else:
                # The probability is equal to 1-d divided by the amount of links
                model[key] = (1 - damping_factor) / len(corpus)
    # If the page has no links
    else:
        # The probability is equal for every link in corpus
        for key in corpus:
            model[key] = 1 / len(corpus)
    
    return model
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Create dict for the results
    results = dict()
    # Populate with each key in corpus at set to 0
    for key in corpus:
        results[key] = 0

    # First sample
    sample = random.choices(list(corpus.keys()))
    # Add 1 to that link PageRank counter
    results[sample[0]] = 1

    # Loop n times minus the first sample
    counter = n - 1
    while counter > 0:
        # Get the transition model
        transition = transition_model(corpus, sample, damping_factor)
        # Choose another sample from that transition model
        sample = random.choices(list(corpus.keys()), transition.values(), k=1)
        # Add 1 to the sample PageRank counter
        results[sample[0]] += 1
        counter -= 1

    # Loop over the results
    for key in results:
        # Adjust the results to a percentage
        results[key] = results[key] / n

    return results


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Create results dict
    results = dict()

    # Keep a copy of previous results
    prev_results = dict()

    # Populate with every link starting with PR equal to 1 divided by the amount of links in corpus
    for key in corpus:
        prev_results[key] = 1 / len(corpus)

    breaker = 0
    # Loop until return
    while True:        
        # Loop over every link
        for key in prev_results:
            # Calculate the sum
            sum = 0
            # Loop over every link
            for link in corpus:
                # If that page links to key
                if key in corpus[link]:
                    # The sum is equal to the PR of that link divided by the amount of links the link has
                    sum += prev_results[link] / len(corpus[link])
                # If that page has no links
                if len(corpus[link]) == 0:
                    # The sum is equal to the PR of that link divided by the amount of links in the corpus
                    sum += prev_results[link] / len(corpus)
            
            # Update PR
            results[key] = ((1 - damping_factor) / len(corpus)) + (damping_factor * sum)
            # Truncate the numbers to 4 decimals
            x = math.trunc(results[key] * (10**4)) / 10**4
            y = math.trunc(prev_results[key] * (10**4)) / 10**4
            # Check if the difference between new PR and previous one are less than 0.001
            if math.trunc(abs(x - y) * 10**3) <= 1:
                # Add to the counter to check if to break
                breaker += 1
        # If the counter is equal to the length of the corpus, meaning, if every PR(key) of the corpus has a difference of 0.001 or less
        # with its previous PR
        if breaker == len(corpus):
            #print(counter)
            return results
        else:
        # If not every key has a difference between its current PR and the previous equal or less to 0.001, then reset the counter    
            breaker = 0
            prev_results = results.copy()
            
           

if __name__ == "__main__":
    main()
