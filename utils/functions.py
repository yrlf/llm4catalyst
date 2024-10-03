import json
from datetime import datetime
from typing import List, Tuple, Dict, Set
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from langchain_openai import ChatOpenAI
<<<<<<< Updated upstream
=======
#from langchain.llms import OpenAI
>>>>>>> Stashed changes
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





class CauseEffectPairs(BaseModel):
    pairs: str = Field(
        default=None, 
        description="Identify factors related to electrocatalyst performance. Your answer should be a list of tuples, where each tuple (A, B) indicates A causes B or A is related to B. For example, ('temperature', 'reaction rate') indicates that temperature causes the reaction rate."
    )
    positive: Optional[bool] = Field(
        default=None, 
        description="Whether the relationship between the factors and the performance of the new materials is positive or negative."
    )
    
class Causality(BaseModel):
    introduction: Optional[CauseEffectPairs] = Field(default=None, description="List of factors related to materials properties described in the text.")

factor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant to an electrocatalyst material scientist. Carefully read each sentence provided from an abstract and introduction, from a paper related to a new material for electrocatalysis. Your task is (1) to identify the factors related to materials properties described in the text. and (2) evaluate the causal relationships between the factors and the performance of the new materials. Note acedmemic writting usually starts with a claim, followed by supporting evidence. You dont have to generate a triplet to if there is no causal relationship between the factors and the performance of the new materials."),
    ("human", "{introduction}"),
])



# LLM to deduplicate the causal pairs
class SimilarFactors(BaseModel):
    is_similar: bool = Field(False, description="Determine whether the two factors are similar or not. An example of similar factors is 'temperature' and 'reaction temperature', another example is 'ORR activity' and 'reaction activity'.")

similar_factor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant to an electrocatalyst material scientist. Your task is to evaluate whether two factors are similar. Similarity indicates that the factors are effectively duplicated, with slight variations that do not change their meaning. Differences in nature do not count as similarity."),
    ("human", "Please evaluate the following factors to determine if they are similar or describe the same concept:\n\nFactor 1: {factor1}\nFactor 2: {factor2}\n\nAre these factors similar?")
])

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

def extract_factors(abstract, introduction, factor_prompt=factor_prompt, chunk_size=1000, chunk_overlap=100):
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    text_splitter = TokenTextSplitter(
        # Controls the size of each chunk
        chunk_size= chunk_size,
        # Controls overlap between chunks
        chunk_overlap= chunk_overlap
    )

    splitted_content= text_splitter.split_text(abstract+introduction)
    

    merged_response=[]
    for i in range(len(splitted_content)):
        runnable = factor_prompt | llm.with_structured_output(schema=Causality)
        response = runnable.invoke({"introduction": {splitted_content[i]}})
        merged_response.append(response.introduction)

    causal_pairs = []
    for response in merged_response:
        if ',' not in response.pairs or len(response.pairs.split(","))!=2:
            continue
        cause, effect = response.pairs.split(",")
        
        direction = response.positive
        causal_pairs.append((cause, effect, direction))
    
    # 获取因子集合
    factors = set()
    for A, B, _ in causal_pairs:
        factors.add(A)
        factors.add(B)
    # 合并相似的因子
    merged_factors = merge_factors(factors, llm)
        
    # 更新因果关系对
    updated_causal_pairs = []
    for A, B, positive in causal_pairs:
        updated_A = merged_factors.get(A, A)
        updated_B = merged_factors.get(B, B)
        updated_causal_pairs.append((updated_A, updated_B, positive))
    final_factors = set()
    for A, B, _ in updated_causal_pairs:
        final_factors.add(A)
        final_factors.add(B)
    print(f"extracted factors: {final_factors}")
    return list(final_factors)


class Measurement(BaseModel):
<<<<<<< Updated upstream
    type: Optional[str] = Field(default=None, description="Type or name of the property measured, such as onset potential, diameter, particle size, conductivity, mass loading, etc.")
    associated_factor: Optional[str] = Field(default=None, description="Associated factor with the measurement, confined to a predefined factor set. For example, ORR activity for electrocatalyst performance.")
    value: float = Field(default=None, description="Property value with units, such as 100 or 0.1, do not include unit here")
    unit: Optional[str] = Field(default=None, description="Unit of the property value, such as V, mA/cm², etc.")
    conditions: Optional[str] = Field(default=None, description="Measurement conditions and techniques/methods, such as in 0.1 M KOH, cyclic voltammetry, galvanostatic charge/discharge, LSV, etc.")
    reaction_type: Optional[str] = Field(default=None, description="Type of reaction, if related to catalyst performance. Choose from ORR, OER, HER. Otherwise, keep it as NA.")
    evidence: int = Field(default=-1, description="The index of the sentence in the text providing evidence for the measurement.")
=======
    value: Optional[str] = Field(default=None, description="The value of the property.")
    unit: Optional[str] = Field(default="NA", description="The unit for the property value.")
    conditions: Optional[str] = Field(default="NA", description="Measurement conditions.")
    reaction_type: Optional[str] = Field(default="NA", description="Type of reaction.")
    evidence: Optional[int] = Field(default=None, exclude=True)


class MaterialProperties(BaseModel):
    chemical_composition: Measurement = Field(default_factory=Measurement, description="Chemical composition as a string.")
    doping: Measurement = Field(default_factory=Measurement, description="Doping element and concentration as a string.")
    electronegativity: Measurement = Field(default_factory=Measurement, description="Electronegativity of the elements.")
    xps: Measurement = Field(default_factory=Measurement, description="XPS peak positions in eV.")
    raman: Measurement = Field(default_factory=Measurement, description="Raman shift in cm⁻¹.")
    ftir: Measurement = Field(default_factory=Measurement, description="FTIR absorption peaks in cm⁻¹.")
    xrd: Measurement = Field(default_factory=Measurement, description="XRD peak positions in degrees.")
>>>>>>> Stashed changes

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
    
    # 文本切割器
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_content = text_splitter.split_text(main_text)
    
    # 用于存储索引和文本
    indexed_content = {i: content for i, content in enumerate(splitted_content) if content.strip()}
    print(f"Length of indexed content: {len(indexed_content)}")

    # 读取 JSON 中的提取示例
    with open('prompts/extraction_example.json', 'r') as file:
        example_text = file.read()

    # LLM 提取的提示
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant for a material scientist specialized in electrocatalysts. Carefully read each sentence provided from the experimental section. Your task is to extract properties in the specified format. Only extract numeric values and their units; do not extract descriptive texts."),
        ("system", "1. Identify if the sentence describes a property of either the main topic {main_topic} or common benchmark catalysts such as Pt, RuO2, Pt/C."),
        ("system", "2. Extract the material name and the property with numeric values and their units, including the specified conditions, following the example format provided: {example_text}."),
        ("system", "3. Do not include an 'evidence' field in your output."),
        ("human", "The paper is as follows: {text}. Please extract the material name and its properties.")
    ])

    # LLM 相关性判断的提示
    relevant_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant to an electrocatalyst material scientist. Your task is to evaluate whether the text is describing properties of materials that are relevant to either the main material ({main_topic}) or common benchmark catalysts such as Pt, RuO2, Pt/C."),
        ("human", "Is the following text relevant to the main material {main_topic} or common benchmark catalysts such as Pt, or materials associated with Pt?\n\nText: {text}")
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
    
<<<<<<< Updated upstream


class ToBeRemoved(BaseModel):
    is_redundant: bool = Field(default=False, description="Whether the extracted information is redundant or irrelevant.")

class isRelevantText(BaseModel):
    
    is_relevant: bool = Field(False, description="Whether the text is relevant or not to the main materials or common benchmark such as Pt, or benchmark materials associated with Pt.")


# 判断材料名称相似性函数
def are_materials_similar(name1: str, name2: str, llm) -> bool:
    similar_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant to an electrocatalyst material scientist. Your task is to evaluate whether the two materials are similar or describe the same material. Note that similarity mainly indicates some duplicated (slightly different but similar) pairs caused by LLM extraction/summary. Slight different in material's nature does NOT count as similarity. For instance, CNT-5 and CNT-10 are NOT considered similar."),
        ("human", "Are the materials {name1} and {name2} similar or describe the same material?")
    ])
    
    runnable = similar_prompt | llm.with_structured_output(schema=SameNames)
    response = runnable.invoke({
        "name1": name1,
        "name2": name2
    })

    return response.is_similar


# 合并相似的材料名称
def merge_material_names(materials: List[Material], llm) -> Dict[str, str]:
    uf = UnionFind()
    material_names = [material.material_name for material in materials if material.material_name]


    for i in range(len(material_names)):
        for j in range(i + 1, len(material_names)):
            #print(f"checking similarity between {material_names[i]} and {material_names[j]}")
            if are_materials_similar(material_names[i], material_names[j], llm):
                uf.union(material_names[i], material_names[j])
                print(f"merged {material_names[i]} and {material_names[j]}")
            else:
                print(f"not similar {material_names[i]} and {material_names[j]}")

    merged = {}
    for name in material_names:
        root = uf.find(name)
        if name != root:
            merged[name] = root
    return merged


# 合并属性列表
def merge_properties(properties1: List[Measurement], properties2: List[Measurement]) -> List[Measurement]:
    return properties1 + properties2

# 去重属性
def deduplicate_properties(properties: List[Measurement]) -> List[Measurement]:
    unique_properties = {}
    for prop in properties:
        prop_key = (prop.type, prop.associated_factor, prop.value, prop.conditions)
        if prop_key not in unique_properties:
            unique_properties[prop_key] = prop
    return list(unique_properties.values())

# 更新材料名称和合并属性
def update_material_names(data: Data, merged_names: Dict[str, str], factors) -> Data:
    # 手动更新材料名称
    for material in data.materials:
        if material.material_name in merged_names:
            material.material_name = merged_names[material.material_name]
    
    # 合并相同名称的材料属性
    merged_materials = {}
    for material in data.materials:
        if material.material_name not in merged_materials:
            merged_materials[material.material_name] = Material(
                material_name=material.material_name,
                properties=material.properties,
                benchmark=material.benchmark
            )
        else:
            merged_materials[material.material_name].properties = merge_properties(
                merged_materials[material.material_name].properties,
                material.properties
            )
            merged_materials[material.material_name].benchmark = (
                merged_materials[material.material_name].benchmark or material.benchmark
            )
    
    # 去重属性
    for material in merged_materials.values():
        material.properties = deduplicate_properties(material.properties)
    
    return Data(materials=list(merged_materials.values()))


def extract_properties(main_text, factors, main_topic, window_size, chunk_size, chunk_overlap):
    llm1 = ChatOpenAI(temperature=0, model="gpt-4-turbo") # for extraction
    llm2 = ChatOpenAI(temperature=0, model="gpt-4-turbo") # for reflection
    text_splitter = TokenTextSplitter(
        # Controls the size of each chunk
        chunk_size= chunk_size,
        # Controls overlap between chunks
        chunk_overlap= chunk_overlap
    )

    splitted_content= text_splitter.split_text(main_text)
    
    # index the content
    indexed_content = {}
    j = 0
    for i in range(len(splitted_content)):
        if splitted_content[i] == "":
            continue
        indexed_content[j] = splitted_content[i]
        j+=1
    
    print(f"length of indexed content: {len(indexed_content)}")

    extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot for a material scientist specialized in electrocatalyst. Carefully read each sentence provided from the experimental section. Your task is to extract information in the specified format. Only extract numeric values and their units; do not extract descriptive texts. If a property does not have a numeric value, skip it. If there are multiple materials related to the main topic, extract all of them. Follow these steps:"),
    ("system", "1. Identify if the sentence describes a property of either the main topic {main_topic} or common benchmark catalyst such as Pt, RuO2, Pt/C, anything that is associated with Pt, etc. <IMPORTANT!> Do not forget to include benchmark catalyst in your extraction"),
    ("system", "2. Check if the property can be associated with one or more factors in {factors}."),
    ("system", "3. Extract the material name and the property with numeric values and their units, including the specified conditions and evidence. Do not extract any descriptive text."),
    ("system", "4. Skip the sentence if it does not contain numeric values or if it is not relevant."),
    ("human", "The paper is as follows: {text} and note the text index is {index} (index will be used to trace the original sentence). Please extract the material name and its properties."),
])

    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot for a material scientist specialized in electrocatalyst. Your task is to review the extracted material properties and revise them to ensure they are concise and relevant. Follow these steps:"),
        ("system", "1. Scrutinize the original text to understand the context and details."),
        ("system", "2. Ensure the extracted information is concise and relevant to the main topic {main_topic} or benchmark materials (e.g., Pt as a common benchmark for catalyst activity comparison)."),
        ("system", "3. Set the 'benchmark' attribute of the Material object correctly if the material is a benchmark."),
        ("system", "4. <IMPORTANT!> Exclude any properties related to materials characterization techniques such as FTIR, Raman, UV-vis, XPS, XRD spectrums, or measurements containing non-numeric values. These should not be included in the revised response."),
        ("system", "5. Ensure the revised properties are accurate and formatted correctly."),
        ("human", "The original text is as follows: {original_text}. The extracted properties are as follows: {extracted_properties}. Please revise the properties accordingly.")
    ])

    relevant_prompt = ChatPromptTemplate.from_messages([(
        "system", "You are a helpful assistant to an electrocatalyst material scientist. Your task is to evaluate whether the text is describing some properties of materials that are relevant or not to either the main materials (i.e, {main_topic}) or common benchmark catalyst such as Pt, RuO2, Pt/C, anything that is associated with Pt, etc.. <IMPORTANT!> Do not forget to include benchmark catalyst in your extraction"),
        ("human", "Is the text {text} relevant to the main materials {main_topic} or common benchmark such as Pt, or benchmark materials associated with Pt?")
    ])

    relevant_runnable = relevant_prompt | llm1.with_structured_output(schema=isRelevantText)
    extract_runnable = extract_prompt | llm1.with_structured_output(schema=Data)
    refine_runnable = refine_prompt | llm2.with_structured_output(schema=Data)
    # 分段处理内容并合并结果
    final_data = Data(
        materials=[]
    )
    for i in range(len(indexed_content)):
        is_relevant_response = relevant_runnable.invoke({"main_topic": main_topic, "text": indexed_content[i]})

        if is_relevant_response.is_relevant == False:
            print(f"Skipping: {indexed_content[i]}")
            continue
        window = window_size
        # consolidate the text for extraction (current index, index-1 and index -..window)
        text_for_extraction = ""
        start_index = max(0, i - window)
        for j in range(start_index, i + 1):
            text_for_extraction += indexed_content[j]
        if text_for_extraction:
            print(f"Extracting: {text_for_extraction[0:20]} ...")
        extract_response = extract_runnable.invoke({
            "text": text_for_extraction,
            "factors": list(factors),
            "main_topic": main_topic,
            "index": i
        })
        
        # 将提取的结果进行精简
        refine_response = refine_runnable.invoke({
            "original_text": text_for_extraction,
            "extracted_properties": json.dumps([material.dict() for material in extract_response.materials]),
            "main_topic": main_topic,
        })
        
        
        #print(f"refined\n{refine_response}")
        for material in refine_response.materials:
            for property in material.properties:
                property.evidence = i
            existing_material = next((m for m in final_data.materials if m.material_name == material.material_name), None)
            if existing_material:
                existing_material.properties.extend(material.properties)
            else:
                final_data.materials.append(material)
   
    extracted_data = Data(**json.loads(final_data.json()))
    merged_material_names = merge_material_names(extracted_data.materials, llm1)
    updated_data = update_material_names(extracted_data, merged_material_names, factors)

    extracted_results = Paper(factor=",".join(factors), materials=updated_data.materials, doi="example_doi")

    return extracted_results, indexed_content
=======
>>>>>>> Stashed changes
