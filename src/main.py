#!/usr/bin/env python3
# ArXiv论文追踪与分析器

import os
import arxiv
import datetime
from pathlib import Path
import openai
import time
import logging
import sys
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from jinja2 import Template

# 配置requests允许重定向
requests.adapters.DEFAULT_RETRIES = 5
requests_session = requests.Session()
requests_session.max_redirects = 10

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
# 支持多个收件人邮箱，用逗号分隔
EMAIL_TO = [email.strip() for email in os.getenv("EMAIL_TO", "").split(",") if email.strip()]

PAPERS_DIR = Path("./papers")
CONCLUSION_FILE = Path("./conclusion.md")
CATEGORIES = ["cs.CE", "cs.AI", "cs.CV", "cs.DS", "cs.NI", "cs.SY", "cs.SI", "cs.CR"]
MAX_PAPERS = 20  # 设置为1以便快速测试

# 配置OpenAI API用于DeepSeek
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com/v1"

# 如果不存在论文目录则创建
PAPERS_DIR.mkdir(exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")
logger.info(f"分析结果将写入: {CONCLUSION_FILE.absolute()}")

def get_recent_papers(categories, max_results=MAX_PAPERS):
    """获取最近5天内发布的指定类别的论文"""
    # 计算最近5天的日期范围
    today = datetime.datetime.now()
    five_days_ago = today - datetime.timedelta(days=2)
    
    # 格式化ArXiv查询的日期
    start_date = five_days_ago.strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    
    logger.info(f"日期范围: {five_days_ago} 到 {today}")
    
    # 创建查询字符串，搜索最近5天内发布的指定类别的论文
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    date_range = f"submittedDate:[{start_date}000000 TO {end_date}235959]"
    query = f"({category_query}) AND {date_range}"
    
    try:
        # 直接使用requests访问API
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results
        }
        
        response = requests_session.get(base_url, params=params)
        response.raise_for_status()
        
        # 解析XML响应
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.text)
        entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
        logger.info(f"找到{len(entries)}篇符合条件的论文")
        
        # 定义Paper和Author类
        class Paper:
            def __init__(self, title, published, categories, entry_id, authors, summary):
                self.title = title
                self.published = published
                self.categories = categories
                self.entry_id = entry_id
                self.authors = authors
                self.summary = summary
                self.relevance_score = 0  # 用于后续相关度评分
                self.relevance_stars = ""
                
            def get_short_id(self):
                # 从entry_id中提取短ID
                return entry_id.split('/')[-1].split('v')[0]
                
            def download_pdf(self, filename):
                # 构建PDF URL并下载
                pdf_url = f"https://arxiv.org/pdf/{self.get_short_id()}.pdf"
                logger.info(f"正在下载PDF: {pdf_url} 到 {filename}")
                pdf_response = requests_session.get(pdf_url)
                pdf_response.raise_for_status()
                with open(filename, 'wb') as f:
                    f.write(pdf_response.content)
        
        class Author:
            def __init__(self, name):
                self.name = name
        
        # 转换XML条目为Paper对象
        paper_objects = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in entries:
            # 提取必要信息
            title = entry.find('atom:title', ns).text.strip()
            published_str = entry.find('atom:published', ns).text
            published = datetime.datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            categories = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]
            entry_id = entry.find('atom:id', ns).text
            summary = entry.find('atom:summary', ns).text.strip()
            
            # 提取作者信息
            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name = author_elem.find('atom:name', ns).text
                authors.append(Author(name))
            
            # 创建Paper对象
            paper = Paper(title, published, categories, entry_id, authors, summary)
            paper_objects.append(paper)
        
        return paper_objects
        
    except Exception as e:
        logger.error(f"获取论文失败: {str(e)}", exc_info=True)
        return []

def download_paper(paper, output_dir):
    """将论文PDF下载到指定目录"""
    pdf_path = output_dir / f"{paper.get_short_id().replace('/', '_')}.pdf"
    
    # 如果已下载则跳过
    if pdf_path.exists():
        logger.info(f"论文已下载: {pdf_path}")
        return pdf_path
    
    try:
        logger.info(f"正在下载: {paper.title}")
        paper.download_pdf(filename=str(pdf_path))
        logger.info(f"已下载到 {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"下载论文失败 {paper.title}: {str(e)}")
        return None

def analyze_paper_with_deepseek(paper):
    """使用DeepSeek API分析论文摘要（减少token消耗）"""
    try:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        # 使用摘要而非PDF内容，减少token消耗
        # 只保留必要信息，确保提示词简洁高效
        prompt = f"""
        论文标题: {paper.title}
        作者: {', '.join(author_names)}
        类别: {', '.join(paper.categories)}
        发布时间: {paper.published}
        
        论文摘要:
        {paper.summary}
        
        请分析这篇研究论文并提供：
        1. 简明摘要（3-5句话）
        2. 主要贡献和创新点
        3. 研究方法概述
        4. 实验结果简述
        5. 潜在影响
        
        请使用中文回答，保持简洁。
        """
        
        logger.info(f"正在分析论文: {paper.title}")
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专门总结和分析学术论文的研究助手。请使用中文回复，保持简洁明了。"},
                {"role": "user", "content": prompt},
            ]
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"论文分析完成: {paper.title}")
        return analysis
    except Exception as e:
        logger.error(f"分析论文失败 {paper.title}: {str(e)}")
        return f"**论文分析出错**: {str(e)}"

def get_domain_keywords():
    """获取用户输入的自定义领域关键词"""
    try:
        # 从环境变量读取自定义关键词，支持通过GitHub secrets设置
        custom_keywords = os.getenv("DOMAIN_KEYWORDS", "")
        if custom_keywords.strip():
            logger.info(f"使用GitHub secrets中的自定义领域关键词")
            return custom_keywords.strip()
        else:
            # 如果没有设置，使用默认关键词
            default_keywords = "machine learning, deep learning, artificial intelligence, neural network"
            logger.info(f"使用默认领域关键词: {default_keywords}")
            return default_keywords
    except Exception as e:
        logger.error(f"获取领域关键词出错: {str(e)}")
        return "machine learning, deep learning, artificial intelligence, neural network"

def rank_papers_with_deepseek(papers, domain_keywords):
    """使用DeepSeek API根据领域关键词对论文进行相关度排序"""
    logger.info(f"使用DeepSeek API对{len(papers)}篇论文进行相关度排序")
    
    try:
        # 准备论文数据，格式化为JSON字符串以便API处理
        papers_data = []
        for i, paper in enumerate(papers, 1):
            # 从Author对象中提取作者名
            author_names = [author.name for author in paper.authors]
            
            papers_data.append({
                "id": i,
                "title": paper.title,
                "summary": paper.summary,
                "authors": author_names,
                "categories": paper.categories
            })
        
        # 构建提示词，让DeepSeek API进行排序
        prompt = f"""
        请根据以下研究领域关键词，对提供的学术论文按照相关度进行降序排序：
        
        研究领域关键词: {domain_keywords}
        
        论文列表:
        {papers_data}
        
        请返回排序后的论文ID列表，格式为[1, 3, 2, ...]。
        同时为每篇论文提供1-5星的相关度评级和简短理由。
        请确保回答格式为:
        
        排序结果:
        [排序后的ID列表]
        
        相关度评级:
        论文1: ⭐⭐⭐⭐⭐ - 理由
        论文2: ⭐⭐⭐ - 理由
        ...
        """
        
        logger.info("正在调用DeepSeek API进行论文排序")
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专门进行学术论文相关度分析和排序的研究助手。请根据用户提供的研究领域关键词，对论文进行精准排序。"},
                {"role": "user", "content": prompt},
            ]
        )
        
        analysis_result = response.choices[0].message.content
        logger.info("DeepSeek API排序完成")
        
        # 解析排序结果
        # 提取排序后的ID列表
        import re
        ranking_match = re.search(r'排序结果:\s*\[(.*?)\]', analysis_result, re.DOTALL)
        if ranking_match:
            # 解析排序后的ID
            rank_str = ranking_match.group(1).strip()
            if rank_str:
                sorted_indices = [int(id.strip()) - 1 for id in rank_str.split(',')]  # 转换为0-index
                # 创建排序后的论文列表
                ranked_papers = [papers[i] for i in sorted_indices]
            else:
                # 如果解析失败，返回原始列表
                ranked_papers = papers
        else:
            # 如果解析失败，返回原始列表
            ranked_papers = papers
        
        # 提取并设置相关度评级
        rating_matches = re.findall(r'论文(\d+):\s*(\S+)\s*-\s*(.+)', analysis_result)
        for paper_id, stars, reason in rating_matches:
            idx = int(paper_id) - 1  # 转换为0-index
            if 0 <= idx < len(papers):
                papers[idx].relevance_stars = stars
                # 从星级转换为分数
                stars_count = stars.count('⭐')
                papers[idx].relevance_score = stars_count * 20
        
        # 为没有评级的论文设置默认值
        for paper in papers:
            if not hasattr(paper, 'relevance_score') or paper.relevance_score == 0:
                paper.relevance_score = 50
                paper.relevance_stars = "⭐⭐⭐"
        
        logger.info(f"已完成论文排序，共{len(ranked_papers)}篇")
        for i, paper in enumerate(ranked_papers[:5], 1):
            logger.info(f"Top-{i}: {paper.title} ({paper.relevance_score:.1f}分, {paper.relevance_stars})")
        
        return ranked_papers
    except Exception as e:
        logger.error(f"使用DeepSeek API排序论文失败: {str(e)}")
        # 出错时返回原始列表，但设置默认相关度
        for paper in papers:
            paper.relevance_score = 50
            paper.relevance_stars = "⭐⭐⭐"
        return papers

def write_to_conclusion(papers_analyses):
    """将分析结果写入conclusion.md，包含相关度信息"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 创建或追加到结果文件
    with open(CONCLUSION_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## ArXiv论文 - 最近5天 (截至 {today})\n\n")
        
        for paper, analysis in papers_analyses:
            # 从Author对象中提取作者名
            author_names = [author.name for author in paper.authors]
            
            f.write(f"### {paper.title}\n")
            f.write(f"**作者**: {', '.join(author_names)}\n")
            f.write(f"**类别**: {', '.join(paper.categories)}\n")
            f.write(f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n")
            f.write(f"**链接**: {paper.entry_id}\n")
            f.write(f"**相关度**: {paper.relevance_stars} (相关度分数: {paper.relevance_score:.1f})\n\n")
            f.write(f"{analysis}\n\n")
            f.write("---\n\n")
    
    logger.info(f"分析结果已写入 {CONCLUSION_FILE}")

def format_email_content(papers_analyses):
    """格式化邮件内容，只包含当天分析的论文，突出显示相关度"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    content = f"## 今日ArXiv论文分析报告 ({today})\n\n"
    content += f"### 按相关度排序的Top论文\n\n"
    
    for paper, analysis in papers_analyses:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        content += f"### {paper.title}\n"
        content += f"**作者**: {', '.join(author_names)}\n"
        content += f"**类别**: {', '.join(paper.categories)}\n"
        content += f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n"
        content += f"**链接**: {paper.entry_id}\n"
        content += f"**相关度**: {paper.relevance_stars} (相关度分数: {paper.relevance_score:.1f})\n\n"
        content += f"{analysis}\n\n"
        content += "---\n\n"
    
    return content

def delete_pdf(pdf_path):
    """删除PDF文件"""
    try:
        if pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"已删除PDF文件: {pdf_path}")
        else:
            logger.info(f"PDF文件不存在，无需删除: {pdf_path}")
    except Exception as e:
        logger.error(f"删除PDF文件失败 {pdf_path}: {str(e)}")

def send_email(content):
    """发送邮件，支持多个收件人"""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]) or not EMAIL_TO:
        logger.error("邮件配置不完整，跳过发送邮件")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = ", ".join(EMAIL_TO)
        msg['Subject'] = f"ArXiv论文分析报告 - {datetime.datetime.now().strftime('%Y-%m-%d')}"

        # 使用HTML模板
        html_template = """
        <html>
        <head>
            <meta charset=\"UTF-8\">
            <style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;line-height:1.6;max-width:1000px;margin:0 auto;padding:20px;background-color:#f5f5f5;}.container{background-color:white;padding:30px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;margin-bottom:30px;}h2{color:#34495e;margin-top:40px;padding-bottom:8px;border-bottom:1px solid #eee;}h3{color:#2980b9;margin-top:30px;}.paper-info{background-color:#f8f9fa;padding:15px;border-left:4px solid #3498db;margin-bottom:20px;}.paper-info p{margin:5px 0;}.paper-info strong{color:#2c3e50;}a{color:#3498db;text-decoration:none;}a:hover{text-decoration:underline;}hr{border:none;border-top:1px solid #eee;margin:30px 0;}.section{margin-bottom:20px;}.section h4{color:#2c3e50;margin-bottom:10px;}pre{background-color:#f8f9fa;padding:15px;border-radius:4px;overflow-x:auto;}code{font-family:Consolas,Monaco,'Courier New',monospace;background-color:#f8f9fa;padding:2px 4px;border-radius:3px;}</style>
        </head>
        <body>
            <div class=\"container\">
                {{ content | replace("###", "<h2>") | replace("##", "<h1>") | replace("**", "<strong>") | safe }}
            </div>
        </body>
        </html>
        """
        
        # 将Markdown格式转换为HTML格式
        content_html = content.replace("\n\n", "<br><br>")
        content_html = content_html.replace("---", "<hr>")
        
        template = Template(html_template)
        html_content = template.render(content=content_html)
        
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"邮件发送成功，收件人: {', '.join(EMAIL_TO)}")
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")

def main():
    logger.info("开始ArXiv论文跟踪与分析")
    
    # 获取最近5天的论文
    papers = get_recent_papers(CATEGORIES, MAX_PAPERS)
    logger.info(f"从最近5天找到{len(papers)}篇论文")
    
    if not papers:
        logger.info("所选时间段没有找到论文。退出。")
        return
    
    # 获取领域关键词（来自GitHub secrets）
    domain_keywords = get_domain_keywords()
    logger.info("已获取领域关键词，准备使用DeepSeek API进行论文排序")
    
    # 使用DeepSeek API按相关度排序论文
    ranked_papers = rank_papers_with_deepseek(papers, domain_keywords)
    
    # 只处理top-5相关论文
    TOP_PAPERS_COUNT = 5
    top_papers = ranked_papers[:TOP_PAPERS_COUNT]
    logger.info(f"将处理Top-{TOP_PAPERS_COUNT}相关论文")
    
    # 处理每篇论文（不再下载PDF，直接使用摘要）
    papers_analyses = []
    for i, paper in enumerate(top_papers, 1):
        logger.info(f"正在分析论文 {i}/{len(top_papers)}: {paper.title}")
        logger.info(f"相关度: {paper.relevance_stars} ({paper.relevance_score:.1f}分)")
        
        # 分析论文（使用摘要而非PDF）
        analysis = analyze_paper_with_deepseek(paper)
        papers_analyses.append((paper, analysis))
        
        # 休眠以避免达到API速率限制
        time.sleep(2)
    
    # 将分析结果写入conclusion.md
    write_to_conclusion(papers_analyses)
    
    # 发送邮件
    email_content = format_email_content(papers_analyses)
    send_email(email_content)
    
    logger.info("ArXiv论文追踪和分析完成")
    logger.info(f"已处理Top-{TOP_PAPERS_COUNT}相关论文，结果已保存至 {CONCLUSION_FILE.absolute()}")

if __name__ == "__main__":
    main()
