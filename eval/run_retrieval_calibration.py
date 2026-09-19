from src.retrieval.retriever import load_index,retrieve_similar

TEST_QUERIES = [
    "my phone screen went completely black and won't respond to anything", 
    "how do I update my payment method for my apple subscription",           
    "recommend me a good pizza place nearby",                                        
    "my apple watch band broke after two days",
]

if __name__=="__main__":
    index = load_index()

    for query in TEST_QUERIES:
        results = retrieve_similar(query,index,top_k=1)
        top_match = results[0]
        print(f"Query: {query}")
        print(f"    Top Similarity: {top_match['similarity']:.4f}")
        print(f"    Closest historical message: {top_match['historical_customer_texts']}")
        print()
