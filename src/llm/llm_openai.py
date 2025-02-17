try:
    import openai
    openai_ok=True
except ImportError:
    openai_ok=False

import asyncio
import base64
import time
import json
from pydantic import BaseModel, create_model
from typing import Optional, List, Literal
from concurrent.futures import ThreadPoolExecutor

def openAI_request(a):
    client=a["client"]
    b64_image=a["b64_image"]
    attrs=a["attrs"]

    prompt="You are an AI that analyses images and provides accurate structured information "
    prompt+="Study the central person in the image and return JSON reponse with the "
    prompt+="keys chosen from the list below."
    #prompt+="Be careful to be very accurate."
    #prompt+="Study the persons head if visible to decide if they have a hat and their hair length."
    prompt+="For the color attributes pick the closest match in the person's clothing color."
    prompt+="Check your responses are accurate."
    prompt+="Try hard to analyse the image even if it somewhat unclear or blurry. If you really can't "
    prompt+="please reply with just 'too blurry'."
    #prompt+="Be accurate; if unsure about an attribute, or if the relevant part of the person "
    #prompt+="is not visible, the answer should be false."


    prompt+="\nKey list:\n"
    for a in attrs:
        if ":" in a:
            a=a.split(":")[1]
        prompt+=a+", "

    DynamicModel=create_model(
        'DynamicModel',
        **{a: (Optional[bool], None) for a in attrs}
    )

    m={
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}",
                    "detail":"low"
                }
            }
        ]
    }

    attempts=2

    for attempt in range(attempts):
        r=""
        refusal=None
        fail=True
        try:
            message = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[m],
                #response_format=DynamicModel,
                temperature=0.0,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0,
                max_tokens=500)
            r=message.choices[0].message.content
            refusal=message.choices[0].message.refusal
            fail="unable" in r or "sorry" in r or "cant" in r or "too blurry" in r
        except Exception as e:
            print(f"OpenAI exception: {e}")

        #print(message)
        #print("\n\n")
        #print( message.choices[0].message)
        #print("\n\n")
        #message.choices[0].message.content

        # 4o-mini 3500 prompt/351 completion Around 3900 tokens used; $0.15/Millon = $6/10K images
        # 4o - 384 prompt/351 completion Around 735 tokens used $2.5/Million : = $18/10K images

        #usage=message.usage
        #print(f"Prompt tokens: {usage.prompt_tokens}")
        #print(f"Completion tokens: {usage.completion_tokens}")
        #print(f"Total tokens: {usage.total_tokens}")
        
        if attempt!=0:
            print(f"OpenAI: result on attempt {attempt}: refusal={refusal} fail={fail} r={r}")

        if refusal is None and fail is False:
            break
        if 'too blurry' in r: # don't retry
            break

    #j=json.loads(r)
    #out=""
    #for m in mappings:
    #    if m[0] in j:
    #        if m[1] in str(j[m[0]]):
    #            out+=m[2]+":true,"
    #    else:
    #        print(f"Error : {m[0]} not in {j}")
    #
    #print(f"j={j} final={out}")
    r=r.replace("null","false")
    return r

class LLMOpenAI:
    def __init__(self):
        assert openai_ok, "Try pip install openai"
        self.client = openai.OpenAI()
        self.num_parallel=512

    def get_batch(self):
        return 2048
    
    def get_max_size(self):
        return 512,512

    def generate_attributes(self, attrs, jpegs):

        b64_images=[]
        for j in jpegs:
            b64_images.append(base64.b64encode(j).decode('utf-8'))
        s=[]
        for b in b64_images:
            a={"client":self.client, "b64_image":b, "attrs":attrs}
            s.append(a)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            responses = list(executor.map(openAI_request, s))
        del s

        return responses
    
    """
    
gpt-4o-mini Feb 8 ---------------------
no_person_visible                  TP= 0.0 FP= 0.0 FN= 1.0 p=0.000 r=0.000 F=0.000 e=o1 
is_male                            TP=14.0 FP= 0.1 FN= 0.0 p=0.994 r=1.000 F=0.997 e=
is_female                          TP= 6.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_wearing_hat_or_head_covering    TP= 6.0 FP= 4.0 FN= 0.0 p=0.600 r=1.000 F=0.750 e=m0 n4 o4 o5 
is_wearing_a_mask_or_face_covering TP= 3.0 FP= 1.0 FN= 0.0 p=0.750 r=1.000 F=0.857 e=m0 
wearing_glasses_or_sunglasses      TP= 3.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
has_facial_hair                    TP= 4.0 FP= 0.1 FN= 0.0 p=0.978 r=1.000 F=0.989 e=
has_shoulder_length_hair           TP= 2.0 FP= 1.0 FN= 1.0 p=0.667 r=0.667 F=0.667 e=n7 p1 
has_buzz_cut_or_bald_head          TP= 0.0 FP= 2.0 FN= 1.0 p=0.000 r=0.000 F=0.000 e=m0 n4 p1 
is_child                           TP= 1.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_adult                           TP=20.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_senior                          TP= 1.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_teen                            TP= 0.0 FP= 0.0 FN= 0.0 p=0.000 r=0.000 F=0.000 e=
has_bag_or_backpack                TP= 7.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_wearing_a_uniform               TP= 1.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
has_long_sleeves                   TP=13.0 FP= 3.0 FN= 0.0 p=0.812 r=1.000 F=0.897 e=m5 n7 o5 
has_visible_tattoos                TP= 2.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_wearing_shorts                  TP= 2.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
is_wearing_bright_colored_clothing TP= 6.0 FP= 0.1 FN= 0.0 p=0.985 r=1.000 F=0.993 e=
is_wearing_a_coat_or_jacket        TP= 7.0 FP= 5.0 FN= 1.0 p=0.583 r=0.875 F=0.700 e=m2 n1 n3 n4 n7 o6 
is_carrying_a_weapon               TP= 1.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
has_a_threatening_posture          TP= 1.0 FP= 1.0 FN= 0.0 p=0.500 r=1.000 F=0.667 e=n1 
has_heavy_build                    TP= 0.0 FP= 3.0 FN= 0.0 p=0.000 r=0.000 F=0.000 e=m3 m4 o5 
is_lying_down                      TP= 2.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
top_is_white_or_light              TP= 3.0 FP= 1.0 FN= 2.0 p=0.750 r=0.600 F=0.667 e=m2 n3 o5 
top_is_black_or_gray_or_dark       TP= 9.0 FP= 3.2 FN= 1.0 p=0.740 r=0.900 F=0.812 e=m2 n1 n3 o5 
top_is_blue_or_purple              TP= 4.0 FP= 0.1 FN= 0.0 p=0.978 r=1.000 F=0.989 e=
top_is_green                       TP= 1.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
top_is_red_or_pink                 TP= 0.0 FP= 1.0 FN= 0.0 p=0.000 r=0.000 F=0.000 e=m0 
top_is_orange_or_beige_or_yellow   TP= 1.0 FP= 2.0 FN= 1.0 p=0.333 r=0.500 F=0.400 e=m0 m2 m4 
bottom_is_white_or_light           TP= 1.0 FP= 0.0 FN= 2.0 p=1.000 r=0.333 F=0.500 e=m0 o3 
bottom_is_black_or_gray_or_dark    TP=10.0 FP= 2.0 FN= 0.0 p=0.833 r=1.000 F=0.909 e=m0 o3 
bottom_is_blue_or_purple           TP= 0.0 FP= 0.0 FN= 1.0 p=0.000 r=0.000 F=0.000 e=n6 
bottom_is_green                    TP= 0.0 FP= 0.0 FN= 0.0 p=0.000 r=0.000 F=0.000 e=
bottom_is_red_or_pink              TP= 0.0 FP= 0.0 FN= 0.0 p=0.000 r=0.000 F=0.000 e=
bottom_is_orange_or_beige_or_yellow TP= 2.0 FP= 0.0 FN= 0.0 p=1.000 r=1.000 F=1.000 e=
---------------------
Gpt-4o-mini Feb 8       Overall              TP=133.0  FP=29.5 FN=11.0  p=0.818 r=0.924 F=0.868
Gpt-4o Feb 8            Overall              TP=117.0  FP=21.3 FN=26.0  p=0.846 r=0.818 F=0.832
gpt-4o-2024-11-20       Overall              TP=130.0  FP=28.4 FN=13.5  p=0.821 r=0.906 F=0.861
gpt-4o-mini-2024-07-18  Overall              TP=133.0  FP=27.5 FN=11.0  p=0.829 r=0.924 F=0.874
gpt-4o-mini (16 Feb 25) Overall              TP=132.0  FP=30.5 FN=12.0  p=0.812 r=0.917 F=0.861


    """