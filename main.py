import pickle
import sklearn  #Required to unpickle scikit-learn objects
from fastapi import FastAPI
from pydantic import BaseModel

#Defining the input data shape
#Pydantic validates the incoming data
class Comment(BaseModel):
    text: str

#Initialize the FastAPI App
app = FastAPI(
    title="Spam & Profanity Detection API",
    description="An API that checks comments for both spam and profanity in a single call.",
    version="1.0.0"
)

#Load models at startup
#We load them once here so we don't reload on every request
try:
    with open("spam_model.pkl", "rb") as f:
        spam_model = pickle.load(f)
    print("Spam model loaded successfully.")

    with open("profanity_model.pkl", "rb") as f:
        profanity_model = pickle.load(f)
    print("Profanity model loaded successfully.")
        
except FileNotFoundError as e:
    print(f"FATAL ERROR: Model file not found. {e}")
    print("API will run, but endpoints will return an error.")
    spam_model = None
    profanity_model = None
except Exception as e:
    print(f"Error loading models: {e}")
    spam_model = None
    profanity_model = None

#Define your API endpoints

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome! The ML API is running. Go to /docs for more info."}


@app.post("/check-comment")
async def check_comment(comment: Comment):
    """
    Checks a comment for BOTH spam and profanity.
    This is the recommended all-in-one endpoint.
    """
    if spam_model is None or profanity_model is None:
        return {"error": "One or more models are not loaded. Check server logs."}

    text_to_check = [comment.text]
    
    #Running Spam Model
    spam_prediction = spam_model.predict(text_to_check)
    spam_probability = spam_model.predict_proba(text_to_check)
    
    is_spam = bool(spam_prediction[0]) 
    spam_confidence = float(spam_probability[0][1]) # Prob of class '1'
    
    #Running Profanity Model
    profanity_prediction = profanity_model.predict(text_to_check)
    profanity_probability = profanity_model.predict_proba(text_to_check)
    
    is_profane = bool(profanity_prediction[0])
    profane_confidence = float(profanity_probability[0][1]) # Prob of class '1'
    
    #Combining and returning results
    return {
        "text": comment.text,
        "spam_check": {
            "is_spam": is_spam,
            "confidence": round(spam_confidence, 4)
        },
        "profanity_check": {
            "is_profane": is_profane,
            "confidence": round(profane_confidence, 4)
        }
    }


#the individual endpoints
@app.post("/check-spam")
async def check_spam(comment: Comment):
    """Checks a comment for spam only."""
    if spam_model is None:
        return {"error": "Spam model is not loaded."}
    
    text_to_check = [comment.text]
    prediction = spam_model.predict(text_to_check)
    probability = spam_model.predict_proba(text_to_check)
    
    is_spam = bool(prediction[0]) 
    confidence = float(probability[0][1])
    
    return {
        "text": comment.text,
        "is_spam": is_spam,
        "confidence": round(confidence, 4)
    }


@app.post("/check-profanity")
async def check_profanity(comment: Comment):
    """Checks a comment for profanity only."""
    if profanity_model is None:
        return {"error": "Profanity model is not loaded."}
    
    text_to_check = [comment.text]
    prediction = profanity_model.predict(text_to_check)
    probability = profanity_model.predict_proba(text_to_check)
    
    is_profane = bool(prediction[0])
    confidence = float(probability[0][1])
    
    return {
        "text": comment.text,
        "is_profane": is_profane,
        "confidence": round(confidence, 4)
    }