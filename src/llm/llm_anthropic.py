import anthropic
import base64
import json
from concurrent.futures import ThreadPoolExecutor

def anthropic_request(a):
    client=a["client"]
    b64_image=a["b64_image"]
    prompt=a["prompt"]

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"Anthropic_request error: {e}")
        return ""

    return message.content[0].text

class LLMAnthropic:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.num_parallel=16

    def get_batch(self):
        return 32

    def generate_attributes(self, attrs, jpegs):

        b64_images=[]
        for j in jpegs:
            b64_images.append(base64.b64encode(j).decode('utf-8'))

        prompt="Study the central person in the image and return JSON reponse with the "
        prompt+="following keys describing image attribue. Each key should have boolean value."
        prompt+="If an attribute is present return true. "
        prompt+="If an attribute is NOT present, or if it cannot be determined, please return false."
        prompt+="For the color attributes, they are asking if those color feature prominently in the"
        prompt+=" top or bottom halves of the persons clothing, respectively.\n"
        prompt+="Key list:\n"

        for a in attrs:
            prompt+=a+", "

        s=[]
        for b in b64_images:
            a={"client":self.client, "prompt":prompt, "b64_image":b}
            s.append(a)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            responses = list(executor.map(anthropic_request, s))

        return responses
