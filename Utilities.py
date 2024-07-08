
import sys,yaml
filepath = "tokens.yaml"

# Read the config file to get tokens/API keys
def get_tokens():
    tokendict={}
    
    with open(filepath,'r') as file:
        tokendict = yaml.safe_load(file)
        #print(tokendict)
    
    return tokendict
