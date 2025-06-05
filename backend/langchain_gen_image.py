# 이미지 생성 서비스 프로세스
# 1. User Prompt
# 2. LLM -> User Prompt 확장
# 3. LLM -> 확장 Prompt 번역(영어) : 영어가 훨씬 잘 나옴
# 4. DALL-E3 -> 이미지 생성

import os
import uuid
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from typing_extensions import TypedDict
import requests
from langgraph.graph import StateGraph, END


# 1. .env 파일을 불러오기
_ = load_dotenv(find_dotenv())
openai_api_key=os.getenv("OPENAI_API_KEY")


# 2. LLM모델 생성
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

# 3. Dall-E-3 모델 생성
client = OpenAI(
    api_key=openai_api_key
)

# 4. LangGraph State 생성
class State(TypedDict):
    prompt : str
    image_url: str

# 5. LangGraph Node 생성

def download_image(url: str, save_path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"이미지 다운로드 완료: {save_path}")
    else:
        print(f"이미지 다운로드 실패: Status Code: {response.status_code}")

# 5-1. 프롬프트 확장 (노드 1)
def refine_prompt(state: State) -> State:
    user_input = state["prompt"]
    response = llm.invoke(f"다음 문장을 이미지 생성용으로 개선하고, 개선한 문장만 출력해줘: {user_input}")
    print(f"확장 프롬프트: {response.content}")
    return {"prompt": response.content}

# 5-2. 프롬프트 번역 (노드 2)
def translate_prompt(state: State) -> State:
    prompt = state["prompt"]
    response = llm.invoke(f"다음 문장을 영어로 번역해줘: {prompt}")
    print(f"번역 프롬프트: {response.content}")
    return {"prompt": response.content}

# 5-3. 이미지 생성 (노드 3)
def generate_image(state: State) -> State:
    prompt = state["prompt"]
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url
    save_path = f"./downloads/{uuid.uuid4()}.png"
    download_image(image_url, save_path)
    return {"prompt": prompt, "image_url": image_url}

# 6. LangGraph 생성
workflow = StateGraph(State)

workflow.add_node("refine_prompt", refine_prompt)
workflow.add_node("translate_prompt", translate_prompt)
workflow.add_node("generate_image", generate_image)

workflow.set_entry_point("refine_prompt") # 시작노드
workflow.add_edge("refine_prompt", "translate_prompt") # 노드 연결
workflow.add_edge("translate_prompt", "generate_image") # 노드 연결
workflow.set_finish_point("generate_image") # 종료노드

graph = workflow.compile()


# 7. LangGraph 실행
if __name__ == "__main__":
    query = input("프롬프트: ")
    result = graph.invoke({"prompt": query, "image_url": ""})
    print(f"이미지 생성 URL: {result['image_url']}")