from transformers import BertTokenizer, BertModel
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like at all."

def get_text_embedding(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    assert encoded_input['input_ids'].shape[1] <= 512, "Your text is too long for BERT"
    output = model(**encoded_input)
    text_embedding = output.last_hidden_state[0,0,:]
    return text_embedding.detach().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


if __name__ == '__main__':
    Fruit = "An apple a day keeps the doctor away"
    Animal = "Curious cats creep, climb, and caper, causing chaotic commotion with their cunning caprices!"
    MySentence = "I like fruit"
    print(f'cos(Fruit, Animal) = {cosine_similarity(get_text_embedding(Fruit), get_text_embedding(Animal))}')
    print(f'cos(Fruit, MySentence) = {cosine_similarity(get_text_embedding(Fruit), get_text_embedding(MySentence))}')
    print(f'cos(Animal, MySentence) = {cosine_similarity(get_text_embedding(Animal), get_text_embedding(MySentence))}')



    Bush = "Today, our fellow citizens, our way of life, our very freedom came under attack in a series of deliberate and deadly terrorist acts."
    Obama = "I stand here today humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors."
    Biden = "This is America's day. This is democracy's day. A day of history and hope, of renewal and resolve."
    print(f'cos(Bush, Obama) = {cosine_similarity(get_text_embedding(Bush), get_text_embedding(Obama))}')
    print(f'cos(Bush, Biden) = {cosine_similarity(get_text_embedding(Bush), get_text_embedding(Biden))}')
    print(f'cos(Obama, Biden) = {cosine_similarity(get_text_embedding(Obama), get_text_embedding(Biden))}')

