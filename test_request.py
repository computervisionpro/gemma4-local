import requests
import base64
from glob import glob
import sys
from pprint import pprint



# path = sys.argv[-1]
# files = glob(f"{path}/*")

# for filename in files:
#     print ('-----'*10)
#     print (filename)
#     file_b64 = base64.b64encode(open(filename, 'rb').read()).decode('utf-8')

URL = 'http://0.0.0.0:8000/chat/local_llm/'
system_prompt = "You are a polite assistant, helping people answer their questions. Keep your answer always within 200 words only."
history = [{"role": "system", "content": system_prompt}]

while True:

    prompt = input("\nAsk Anything:\n\n")
    history.append({"role": "user", "content": prompt})

    req_body = {
        "req_id": "ID_______",
        "query": history,
    }



    resp = requests.post(URL, json=req_body)
    # print (resp.text, '------------------------')
    resp = resp.json()
    pprint(resp)

    if resp.get("success", False):
        response = resp.get("response")
        history.append({"role": "assistant", "content": response})
        

    print('\n\n')

                                                                                                                                            
