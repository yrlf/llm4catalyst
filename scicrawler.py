import re
import os
import csv
from collections import deque
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 通用的文章抓取方法
def scrape_article_common(url, title_xpath, abstract_xpath, maintext_xpath, chrome_driver_path):
    try:
        driver = webdriver.Chrome(service=Service(executable_path=chrome_driver_path))
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, title_xpath)))
        
        title = driver.find_element(By.XPATH, title_xpath).text
        abstract = driver.find_element(By.XPATH, abstract_xpath).text
        maintext = driver.find_element(By.XPATH, maintext_xpath).text
        
        return title, abstract, maintext
    except Exception as e:
        print(f"爬取文章时发生错误: {e}, URL: {url}")
        return None, None, None
    finally:
        driver.quit()

# 定义不同期刊的处理函数
def scrape_jacs_article(url, chrome_driver_path):
    title_xpath = '//*[@id="pb-page-content"]/div/main/article/div[4]/div[1]/div[1]/div/div/h1/span'
    abstract_xpath = '//*[@id="pb-page-content"]/div/main/article/div[4]/div[1]/div[2]'
    maintext_xpath = '//*[@id="pb-page-content"]/div/main/article/div[4]/div[1]/div[3]/div[1]/div'
    return scrape_article_common(url, title_xpath, abstract_xpath, maintext_xpath, chrome_driver_path)

def scrape_nature_catalysis_article(url, chrome_driver_path):
    title_xpath = '//h1[@class="c-article-title"]'
    abstract_xpath = '//div[@id="Abs1-content"]'
    maintext_xpath = '//*[@id="content"]/main/article/div[2]/div[2]'
    return scrape_article_common(url, title_xpath, abstract_xpath, maintext_xpath, chrome_driver_path)

def scrape_angewandte_article(url, chrome_driver_path):
    title_xpath = '//*[@id="article__content"]/div[2]/div/h1'
    abstract_xpath = '//*[@id="section-1-en"]/div'
    maintext_xpath = '//*[@id="article__content"]'
    return scrape_article_common(url, title_xpath, abstract_xpath, maintext_xpath, chrome_driver_path)

def scrape_acs_catalysis_article(url, chrome_driver_path):
    title_xpath = '//h1[@class="article_header-title"]'
    abstract_xpath = '//div[@class="article_abstract-content"]'
    maintext_xpath = '//div[@class="article_content"]'
    return scrape_article_common(url, title_xpath, abstract_xpath, maintext_xpath, chrome_driver_path)

# 另存为文件函数
def save_article_content(title, abstract, maintext, url):
    clean_title = re.sub(r'[^\w\s]', '', title).replace(' ', '_')[:50]

    # 确保 documents 目录存在
    documents_dir = os.path.join(os.getcwd(), 'documents')
    abstract_dir = os.path.join(documents_dir, 'abstract')
    maintext_dir = os.path.join(documents_dir, 'maintext')
    
    os.makedirs(abstract_dir, exist_ok=True)
    os.makedirs(maintext_dir, exist_ok=True)
    
    # 保存摘要到文件
    abstract_filename = os.path.join(abstract_dir, f"{clean_title}_abs.txt")
    with open(abstract_filename, 'w', encoding='utf-8') as file:
        file.write(abstract)
    
    # 保存主内容到文件
    maintext_filename = os.path.join(maintext_dir, f"{clean_title}_maintext.txt")
    with open(maintext_filename, 'w', encoding='utf-8') as file:
        file.write(maintext)
    
    # 记录文章标题和URL到 CSV 文件
    record_filename = os.path.join(documents_dir, 'record.csv')
    record_exists = os.path.isfile(record_filename)
    
    with open(record_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not record_exists:
            writer.writeheader()
        
        writer.writerow({'title': clean_title, 'url': url})

# Google Scholar 爬取相关文章
def search_scholar_and_scrape_related(url, chrome_driver_path, num_related=40):
    """使用 Selenium 从 Google Scholar 获取与提供的 URL 相关的文章链接"""
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)

    related_urls = []

    try:
        # 打开 Google Scholar
        driver.get('https://scholar.google.com/')
        print("成功打开 Google Scholar")
        
        # 在搜索框中输入 URL 并提交搜索
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, 'q'))
        )
        search_box.send_keys(url)
        search_box.submit()
        print(f"已提交搜索：{url}")

        # 等待搜索结果加载完成
        search_results_xpath = '//h3[@class="gs_rt"]/a'
        search_results = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, search_results_xpath))
        )
        print(f"找到 {len(search_results)} 个搜索结果")
        
        # 查找第一个结果的"相关文章"链接
        related_articles_xpath = '//a[contains(@href, "related") and contains(text(), "相关文章")]'
        related_articles_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, related_articles_xpath))
        )
        print("找到'相关文章'链接")
        related_articles_element.click()
        print("已点击'相关文章'链接")

        # 等待相关文章页面加载
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//h3[@class="gs_rt"]/a'))
        )
        print("相关文章页面已加载")

        # 获取相关文章的链接
        related_article_elements = driver.find_elements(By.XPATH, '//h3[@class="gs_rt"]/a')
        for element in related_article_elements[:num_related]:
            related_urls.append(element.get_attribute('href'))
        
        print(f"找到 {len(related_urls)} 个相关文章链接")

        return related_urls
    
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return related_urls
    
    finally:
        driver.quit()

# 根据URL确定期刊的函数
def determine_journal_from_url(url):
    """根据 URL 确定期刊名称"""
    if "sciencedirect.com" in url:
        return "Elsevier"
    elif "pubs.acs.org" in url:
        if "jacs" in url:
            return "Journal of the American Chemical Society"
        elif "nn" in url or "acsnano" in url:
            return "ACS Nano"
        elif "acscatal" in url:
            return "ACS Catalysis"
        else:
            return "ACS"
    elif "onlinelibrary.wiley.com" in url:
        if "adma" in url:
            return "Advanced Materials"
        elif "ange" in url:
            return "Angewandte Chemie"
    elif "science.org" in url:
        if "science" in url:
            return "Science"
        elif "sciadv" in url:
            return "Science Advances"
    else:
        return "Unknown"

# 根据期刊名称调用相应的文章爬取方法
def scrape_article_by_journal(url, journal, chrome_driver_path):
    """根据期刊名称选择合适的爬取函数"""
    if journal.lower() == 'journal of the american chemical society' or journal.lower() == 'jacs':
        return scrape_jacs_article(url, chrome_driver_path)
    elif journal.lower() == 'nature catalysis':
        return scrape_nature_catalysis_article(url, chrome_driver_path)
    elif journal.lower() == 'angewandte chemie' or journal.lower() == 'angewandte':
        return scrape_angewandte_article(url, chrome_driver_path)
    elif journal.lower() == 'acs nano':
        return scrape_acs_catalysis_article(url, chrome_driver_path)
    elif journal.lower() == 'science':
        return scrape_science_article(url, chrome_driver_path)
    elif journal.lower() == 'science advances':
        return scrape_science_advances_article(url, chrome_driver_path)
    elif journal.lower() == 'advanced materials':
        return scrape_acs_catalysis_article(url, chrome_driver_path)
    elif journal.lower() == 'elsevier':
        return scrape_article_common(url, '//span[@class="title-text"]', '//div[@class="abstract"]', '//div[@class="content"]', chrome_driver_path)
    else:
        print(f"Unsupported journal: {journal}")
        return None, None, None


# 读取初始的 txt 文件作为入参
def read_urls_from_txt(file_path):
    """读取 txt 文件中的 URL 列表"""
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]
    return urls

# 主程序入口
chrome_driver_path = "/Users/yangz/Downloads/chromedriver/chromedriver"
txt_file_path = "urls.txt"
csv_path = "documents/record.csv"

# 检查文件是否存在
if not os.path.exists(csv_path):
    # 如果文件不存在，创建它并添加指定的标题
    df = pd.DataFrame(columns=['title', 'url'])
    df.to_csv(csv_path, index=False)
else:
    # 如果文件存在，读取它
    df = pd.read_csv(csv_path)

# 从 txt 文件读取初始 URL
initial_urls = read_urls_from_txt(txt_file_path)

# 使用 set 初始化已访问的 URL
visited = set(df['url'].tolist())
queue = deque(initial_urls)
total_processed = 0
max_total = 10  # 设置爬取的最大数量

while queue and total_processed < max_total:
    current_url = queue.popleft()
    
    if current_url in visited:
        continue
    
    visited.add(current_url)
    
    try:
        # 爬取当前 URL 的内容
        journal = determine_journal_from_url(current_url)
        print(f"正在处理 URL: {current_url}")
        print(f"期刊: {journal}")
        
        title, abstract, maintext = scrape_article_by_journal(current_url, journal, chrome_driver_path)
        if title and abstract and maintext:
            save_article_content(title, abstract, maintext, current_url)
            total_processed += 1
            print(f"成功爬取并保存文章: {title}")
        else:
            print(f"无法爬取文章内容: {current_url}")
        
        # 查找并添加相关 URL
        related_urls = search_scholar_and_scrape_related(current_url, chrome_driver_path, 40)
        print(f"找到 {len(related_urls)} 个相关 URL")
        for related_url in related_urls:
            if related_url not in visited:
                queue.append(related_url)
        
    except Exception as e:
        print(f"处理 {current_url} 时发生错误: {e}")
    
    if total_processed >= max_total:
        print(f"已达到最大处理数量 {max_total}。退出程序。")
        break

print(f"总共处理的 URL 数量: {total_processed}")
