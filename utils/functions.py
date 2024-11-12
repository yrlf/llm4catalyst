import json
from datetime import datetime
from typing import List, Tuple, Dict, Set
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from langchain_openai import ChatOpenAI
#from langchain.llms import OpenAI
import networkx as nx
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import TokenTextSplitter
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain_openai import OpenAI
import nltk
import os
from dotenv import load_dotenv
from typing import Optional
from langchain.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic

load_dotenv("api.env")
api_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


def read_properties(filePath):
    # read properties from file and return a list of properties
    with open(filePath, 'r') as f:
        properties = f.readlines()

    properties = [x.strip() for x in properties]
    return properties

def generate_prompts(propertyFilePath, promptFilePath):
    # Read properties from file
    properties_list = read_properties(propertyFilePath)

    # Read prompt from file
    PROMPTS_FILE = promptFilePath
    with open(PROMPTS_FILE, 'r') as file:
        prompt_template = file.read()

    # Replace the placeholder with actual properties
    prompt = prompt_template.format(properties_list=', '.join(properties_list))

    return prompt


def split_text(text, max_length):
    """
    Split the text into chunks of max_length.
    """
    words = text.split()
    chunks = []
    chunk = ""

    for word in words:
        if len(chunk) + len(word) <= max_length:
            chunk += " " + word
        else:
            chunks.append(chunk)
            chunk = word
    if chunk:
        chunks.append(chunk)

    return chunks


def api_query(api_type, model, prompt, additional_params=None):
    """
    Make an API query to either OpenAI or Hugging Face, depending on the api_type.
    :param api_type: 'openai' or 'huggingface'
    :param prompt: The prompt to be sent to the API.
    :param additional_params: Additional parameters for the API request.
    :return: The API response.
    """
    if additional_params is None:
        additional_params = {}

    # OpenAI API
    if api_type == 'openai':
        response = openai_api_query(prompt, additional_params)
    # Hugging Face API
    elif api_type == 'huggingface':
        response = huggingface_api_query(prompt, additional_params)
    else:
        raise ValueError("Unsupported API type")

    return response

def openai_api_query(prompt, params):
    # Assume implementation for querying OpenAI's API
    # This should include setting up headers, endpoint, body, etc.
    pass

def huggingface_api_query(prompt, params):
    # Assume implementation for querying Hugging Face's API
    # This should include setting up headers, endpoint, body, etc.
    pass





class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX



def read_file(file_path):
    try:
        with open(file_path, 'r') as f:
            text = f.read()
    except:
        print("Error loading file: ", file_path)
    
    return text



class MainTopic(BaseModel):
    main_material_name: Optional[str] = Field (default = None, description="Name of the material, This could be different from conventional represnetation since authors may use their own symbol to represent a newly developed matreials. For example carbon nanotube synthesized at 60 degree may be written as CNT@60")
    chemical_symbol: Optional[str] = Field (default = None, description="Chemical symbol representation of the material")
main_topic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot of a material scientist specialized in electrocatalyst. Carefully read the abstract provided. You need to identify what is the main materials reported in this literature. The paper typically reports a new material or a new combination of materials. You need to extract the full name described as well as any chemical symbols. For example carbon nanotube synthesized at 60 degree may be written as CNT@60. The abstract is as follows: {abstract}"),
    ("human", "{abstract}")
])

def extract_main_topic(abstract, main_topic_prompt=main_topic_prompt,):
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    main_topic_runnable = main_topic_prompt | llm.with_structured_output(schema=MainTopic)
    main_topic_response = main_topic_runnable.invoke({"abstract":{abstract}})
    print(f"main topic response: {main_topic_response}")
    return main_topic_response



class SameNames(BaseModel):
    is_similar: bool = Field(False, description="Determine whether the two material names are same. Same names are caused by LLM extraction and there are only slight difference in names. ")


def are_factors_similar(factor1: str, factor2: str, llm) -> bool:
    runnable = similar_factor_prompt | llm.with_structured_output(schema=SimilarFactors)
    response = runnable.invoke({"factor1": factor1, "factor2": factor2})
    return response.is_similar

# 合并相似的因子
def merge_factors(factors: Set[str], llm) -> Dict[str, str]:
    # print(f"before merge: {factors}")
    uf = UnionFind()
    factors = list(factors)
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            if are_factors_similar(factors[i], factors[j], llm):
                uf.union(factors[i], factors[j])
    
    merged = {}
    for factor in factors:
        root = uf.find(factor)
        if factor != root:
            merged[factor] = root

    return merged


class Measurement(BaseModel):
    #value: Optional[str] = Field(default=None, description="The value of the property.")
    value: Optional[str] = Field(default=None, description="The value of the property as a dictionary.")
    unit: Optional[str] = Field(default="NA", description="The unit for the property value such as nm, eV, cm-1, mV, mA, mg/cm2, %, ppm, etc.")
    conditions: Optional[str] = Field(default="NA", description="Measurement conditions such as temperature, pressure, pH values etc.")
    reaction_type: Optional[str] = Field(default="NA", description="Type of reaction, must be one of the following: HER, OER, ORR, NA")
    evidence: Optional[int] = Field(default=None, exclude=True)

class Spectroscopy(BaseModel):
    value : list[str] = Field(default="[]", description="The value of the occurrence of the spectroscopy peak, must be a list of numbers. Example: [243.1, 244.1, 245.1]")
    unit: Optional[str] = Field(default="NA", description="The unit for the property value such eV, cm-1, etc.")
    conditions: Optional[str] = Field(default="NA", description="Measurement conditions such as temperature, pressure, pH values etc.")
    evidence: Optional[int] = Field(default=None, exclude=True)



class MaterialProperties(BaseModel):
    chemical_composition: Measurement = Field(default_factory=Measurement, description="Chemical composition or chemical formula")
    doping: Measurement = Field(default_factory=Measurement, description="Doping element and concentration as a string.")
    electronegativity: Measurement = Field(default_factory=Measurement, description="Electronegativity of the elements.")
    xps: Measurement = Field(default_factory=Spectroscopy, description="XPS peak positions in eV.")
    raman: Measurement = Field(default_factory=Spectroscopy, description="Raman shift in cm⁻¹.")
    ftir: Measurement = Field(default_factory=Spectroscopy, description="FTIR absorption peaks in cm⁻¹.")
    xrd: Measurement = Field(default_factory=Spectroscopy, description="XRD peak positions in degrees.")
    onset_potential: Measurement = Field(default_factory=Measurement, description="Onset potential in mV.")
    mass_loading: Measurement = Field(default_factory=Measurement, description="Mass loading in mg/cm2.")
    overpotential: Measurement = Field(default_factory=Measurement, description="Overpotential in mV.")
    tafel_slope: Measurement = Field(default_factory=Measurement, description="Tafel slope in mV.")


class Material(BaseModel):
    material_name: Optional[str] = Field(default=None, description="Name of the material.")
    properties: MaterialProperties = Field(default_factory=MaterialProperties, description="Properties of the material.")
    benchmark: bool = Field(default=False, description="")

class Data(BaseModel):
    materials: list[Material] = Field(default_factory=list, description="")
    #results: Optional[str] = Field(default=None, description="")


    # 定义 isRelevantText 的 schema
class isRelevantText(BaseModel):
    is_relevant: bool = Field(False, description="Whether the text is relevant to the main materials or common benchmark catalysts.")


class Paper(Data):
    doi: Optional[str] = Field(default=None, description="")
    factor: Optional[str] = Field(default=None, description="")
    meta: Optional[dict] = Field(default=None, description="")
def extract_properties(main_text, factors, main_topic, window_size, chunk_size, chunk_overlap):
    # 初始化 OpenAI 模型
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    #llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # 文本切割器
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_content = text_splitter.split_text(main_text)
    
    # 用于存储索引和文本
    indexed_content = {i: content for i, content in enumerate(splitted_content) if content.strip()}
    print(f"Length of indexed content: {len(indexed_content)}")

    # # 读取 JSON 中的提取示例
    with open('prompts/extraction_example.json', 'r') as file:
        example_text = file.read()

    # LLM 提取的提示
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant for a material scientist specialized in electrocatalysts. Carefully read each sentence provided from the experimental section. Your task is to extract properties in the specified format. Only extract numeric values and their units; do not extract descriptive texts."),
        ("system", "1. Identify if the sentence describes a property of either the main topic {main_topic}"),
        ("system", "2. Extract the material name and the property with numeric values and their units, including the specified conditions. If it is not a property of the main topic, please ignore it. If the property is not reported, please report it as NA. An example is as follows: {example_text}"),
        ("system", "3. For Raman, FTIR, XPS, and XRD, the value is a list of numbers representing the peak positions. For example, if two peaks are detected at 243.1 cm-1 and 244.1 cm-1, the value should be [243.1, 244.1]. If only one peak is detected, the value should be [243.1]. You should include all the peaks of that material detected in the text."),
        ("system", "4. Do not include an 'evidence' field in your output."),
        ("human", "The paper is as follows: {text}. Please extract the material name and its properties.")
    ])

    # LLM 相关性判断的提示
    relevant_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant to an electrocatalyst material scientist. Your task is to evaluate whether the text is describing properties of materials that are relevant to either the main material ({main_topic})"),
        ("human", "Is the following text relevant to the main material {main_topic}\n\nText: {text}")
    ])

    # 创建 runnables
    relevant_runnable = relevant_prompt | llm.with_structured_output(schema=isRelevantText)
    extract_runnable = extract_prompt | llm.with_structured_output(schema=Data)

    # 初始化最终数据
    final_data = Data(materials=[])

    # 遍历切割后的文本块并进行提取
    for i, text in indexed_content.items():
        # 检查文本是否相关
        is_relevant_response = relevant_runnable.invoke({
            "main_topic": main_topic,
            "text": text
        })

        if not is_relevant_response.is_relevant:
            print(f"Skipping irrelevant text at index {i}: {text[:30]}...")
            continue

        # 准备用于提取的文本（考虑窗口大小）
        window = window_size
        text_for_extraction = ""
        start_index = max(0, i - window)
        for j in range(start_index, i + 1):
            text_for_extraction += indexed_content[j] + " "
        if text_for_extraction:
            print(f"Extracting from text at index {i}: {text_for_extraction[:50]}...")

        # 执行提取
        extract_response = extract_runnable.invoke({
            "text": text_for_extraction,
            "main_topic": main_topic,
            "example_text": example_text
        })

        # 收集提取的材料信息
        if extract_response.materials:
            for material in extract_response.materials:
                # 手动设置每个属性的 evidence 索引
                # material.properties 是 MaterialProperties 的实例
                properties = material.properties
                for prop_name, measurement in properties.__dict__.items():
                    if measurement and isinstance(measurement, Measurement):
                        measurement.evidence = i  # 设置 evidence 索引
                final_data.materials.append(material)

    # 准备最终提取结果
    extracted_results = Paper(
        factor=",".join(factors),
        materials=final_data.materials,
        doi="example_doi"
    )

    return extracted_results, indexed_content



class Paper(Data):
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier (DOI) of the paper.")

    factor: Optional[str] = Field(default=None, description="Factor related to the material properties described in the text.")

    meta: Optional[Dict[str, str]] = Field(default=None, description="Additional metadata about the paper, such as authors, journal, publication date, etc.")
    
