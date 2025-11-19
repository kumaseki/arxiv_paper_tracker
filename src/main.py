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

# 从环境变量获取领域关键词（支持从GitHub Secrets中设置）
DOMAIN_KEYWORDS = os.getenv("DOMAIN_KEYWORDS", "")

PAPERS_DIR = Path("./papers")
CONCLUSION_FILE = Path("./conclusion.md")
CATEGORIES = ["cs.CE", "cs.AI", "cs.CV", "cs.DS", "cs.NI", "cs.SY", "cs.SI", "cs.CR"]
MAX_PAPERS = 50  # 最大论文数量
TOP_PAPERS = 5  # 输出最相关的top-N论文

# 配置OpenAI API用于DeepSeek
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com/v1"

# 如果不存在论文目录则创建
PAPERS_DIR.mkdir(exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")
logger.info(f"分析结果将写入: {CONCLUSION_FILE.absolute()}")

def get_recent_papers(categories, max_results=MAX_PAPERS):
    """获取最近几天内发布的指定类别的论文"""
    # 计算日期范围
    today = datetime.datetime.now()
    five_days_ago = today - datetime.timedelta(days=5)
    
    # 格式化ArXiv查询的日期
    start_date = start_date_obj.strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    
    logger.info(f"日期范围: {five_days_ago} 到 {today}")
    
    # 创建查询字符串，搜索最近5天内发布的指定类别的论文
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    date_range = f"submittedDate:[{start_date}000000 TO {end_date}235959]"
    query = f"({category_query}) AND {date_range}"
    
    try:
        # 使用requests访问API
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
        
        # 创建模拟的Paper对象列表
        class Paper:
            def __init__(self, title, published, categories, entry_id, authors, summary):
                self.title = title
                self.published = published
                self.categories = categories
                self.entry_id = entry_id
                self.authors = authors
                self.summary = summary
                
            def get_short_id(self):
                # 从entry_id中提取短ID
                return self.entry_id.split('/')[-1].split('v')[0]
                
            def download_pdf(self, filename):
                # 构建PDF URL并下载
                pdf_url = f"https://arxiv.org/pdf/{self.get_short_id()}.pdf"
                pdf_response = requests_session.get(pdf_url)
                pdf_response.raise_for_status()
                with open(filename, 'wb') as f:
                    f.write(pdf_response.content)
        
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
            class Author:
                def __init__(self, name):
                    self.name = name
            
            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name = author_elem.find('atom:name', ns).text
                authors.append(Author(name))
            
            # 创建Paper对象
            paper = Paper(title, published, categories, entry_id, authors, summary)
            paper_objects.append(paper)
        
        return paper_objects
        
    except Exception as e:
        logger.error(f"查询论文出错: {str(e)}", exc_info=True)
        return []

def download_paper(paper, output_dir):
    """将论文PDF下载到指定目录"""
    pdf_path = output_dir / f"{paper.get_short_id().replace('/', '_')}.pdf"
    
    # 如果已下载则跳过
    if pdf_path.exists():
        return pdf_path
    
    try:
        paper.download_pdf(filename=str(pdf_path))
        return pdf_path
    except Exception as e:
        logger.error(f"下载论文失败 {paper.title}: {str(e)}")
        return None

def analyze_paper_with_deepseek(pdf_path, paper, domain_keywords=""):
    """使用DeepSeek API分析论文（使用OpenAI 0.28.0兼容格式）"""
    try:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        # 构建prompt，控制token数量
        prompt = f"论文标题: {paper.title}\n"
        prompt += f"作者: {', '.join(author_names)}\n"
        prompt += f"类别: {', '.join(paper.categories)}\n"
        prompt += f"发布时间: {paper.published}\n"
        
        # 如果有领域关键词，添加到prompt中
        if domain_keywords:
            prompt += f"领域关键词: {domain_keywords}\n"
        
        prompt += "\n请以简明扼要的方式分析这篇论文，并回答以下问题:\n"
        prompt += "1. 简明摘要（2-3句话）\n"
        prompt += "2. 主要贡献\n"
        prompt += "3. 研究方法\n"
        prompt += "4. 与给定领域关键词的相关程度（1-5星，1星最低，5星最高）以及简要理由\n"
        prompt += "5. 结论\n"
        prompt += "\n请使用中文，保持简洁，每部分控制在2-3句话。"
        
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专门总结和分析学术论文的研究助手。请使用中文回复。"},
                {"role": "user", "content": prompt},
            ]
        )
        
        analysis = response.choices[0].message.content
        return analysis
    except Exception as e:
        logger.error(f"分析论文失败 {paper.title}: {str(e)}")
        return f"**论文分析出错**: {str(e)}"

def write_to_conclusion(papers_with_scores):
    """将分析结果写入conclusion.md"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 创建或追加到结果文件
    with open(CONCLUSION_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## ArXiv论文 - 最近5天 (截至 {today})\n\n")
        
        # 如果提供了领域关键词，显示出来
        if DOMAIN_KEYWORDS:
            f.write(f"**领域关键词**: {DOMAIN_KEYWORDS}\n\n")
        
        for i, item in enumerate(papers_with_scores, 1):
            # 支持新旧两种格式的数据
            if len(item) == 3:
                paper, analysis, score = item
                # 生成星级显示
                stars = '⭐' * score
            else:
                paper, analysis = item
                stars = '⭐⭐⭐'  # 默认三星
            
            # 从Author对象中提取作者名
            author_names = [author.name for author in paper.authors]
            
            f.write(f"### {i}. {paper.title}\n")
            f.write(f"**作者**: {', '.join(author_names)}\n")
            f.write(f"**类别**: {', '.join(paper.categories)}\n")
            f.write(f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n")
            f.write(f"**链接**: {paper.entry_id}\n")
            f.write(f"**相关性评分**: {stars} ({score}星)\n\n" if len(item) == 3 else "")
            f.write(f"{analysis}\n\n")
            f.write("---\n\n")
    
    logger.info(f"分析结果已写入 {CONCLUSION_FILE}")

def format_email_content(papers_with_scores):
    """格式化邮件内容，只包含当天分析的论文"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    content = f"## 今日ArXiv论文分析报告 ({today})\n\n"
    
    # 如果提供了领域关键词，显示出来
    if DOMAIN_KEYWORDS:
        content += f"**领域关键词**: {DOMAIN_KEYWORDS}\n\n"
    
    # 添加说明文字
    content += "以下是按相关性排序的论文分析，按评分从高到低排列：\n\n"
    
    for i, item in enumerate(papers_with_scores, 1):
        # 支持新旧两种格式的数据
        if len(item) == 3:
            paper, analysis, score = item
            # 生成星级显示
            stars = '⭐' * score
        else:
            paper, analysis = item
            stars = '⭐⭐⭐'  # 默认三星
        
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        content += f"### {i}. {paper.title}\n"
        content += f"**作者**: {', '.join(author_names)}\n"
        content += f"**类别**: {', '.join(paper.categories)}\n"
        content += f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n"
        content += f"**链接**: {paper.entry_id}\n"
        content += f"**相关性评分**: {stars} ({score}星)\n\n" if len(item) == 3 else "\n"
        
        # 确保analysis不为空
        if analysis:
            content += f"{analysis}\n\n"
        else:
            content += "暂无分析结果\n\n"
        
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

        # 使用简单直接的HTML模板，避免模板引擎的复杂替换
        html_start = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;}
                .container {background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
                h1 {color: #2c3e50; font-size: 24px; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 30px;}
                h2 {color: #34495e; font-size: 20px; margin-top: 40px; padding-bottom: 8px; border-bottom: 1px solid #eee;}
                h3 {color: #2980b9; font-size: 18px; margin-top: 30px;}
                p {margin: 10px 0; font-size: 16px;}
                strong {color: #2c3e50;}
                a {color: #3498db; text-decoration: none;}
                a:hover {text-decoration: underline;}
                hr {border: none; border-top: 1px solid #eee; margin: 30px 0;}
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        html_end = """
            </div>
        </body>
        </html>
        """
        
        # 将Markdown格式转换为HTML格式
        content_html = content
        
        # 正确的替换顺序：先替换高级标题(##)，再替换低级标题(###)
        content_html = content_html.replace("## ", "<h1>")
        content_html = content_html.replace("\n## ", "</h1><h1>")
        content_html = content_html.replace("### ", "<h2>")
        content_html = content_html.replace("\n### ", "</h2><h2>")
        
        # 替换粗体
        content_html = content_html.replace("**", "<strong>", 1)
        # 处理粗体的结束标签
        while "**" in content_html:
            content_html = content_html.replace("**", "</strong>", 1)
            content_html = content_html.replace("**", "<strong>", 1)
        
        # 替换换行
        content_html = content_html.replace("\n\n", "<br><br>")
        
        # 替换分割线
        content_html = content_html.replace("---\n", "<hr>")
        
        # 关闭所有开放的标题标签
        content_html = content_html.replace("<h1>", "<h1>")
        content_html = content_html.replace("<h2>", "<h2>")
        if "<h1>" in content_html and "</h1>" not in content_html.split("<h1>")[-1]:
            content_html += "</h1>"
        if "<h2>" in content_html and "</h2>" not in content_html.split("<h2>")[-1]:
            content_html += "</h2>"
        
        # 组合完整的HTML内容
        html_content = html_start + content_html + html_end
        
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"邮件发送成功，收件人: {', '.join(EMAIL_TO)}")
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")

def extract_relevance_score(analysis_text):
    """从分析结果中提取相关性评分（1-5星）"""
    # 尝试从分析文本中提取评分
    import re
    
    # 匹配形如 "相关程度：5星" 或 "4星" 等模式
    score_match = re.search(r'(相关程度|相关性).*?([1-5])星', analysis_text, re.IGNORECASE)
    if score_match:
        return int(score_match.group(2))
    
    # 尝试直接匹配数字+星号
    direct_match = re.search(r'([1-5])\s*星', analysis_text)
    if direct_match:
        return int(direct_match.group(1))
    
    # 默认返回1星（最低相关度）
    return 1

def main():
    # 获取最近5天的论文
    papers = get_recent_papers(CATEGORIES, MAX_PAPERS)
    
    if not papers:
        logger.info("所选时间段没有找到论文。退出。")
        return
    
    logger.info(f"找到{len(papers)}篇论文，使用领域关键词: '{DOMAIN_KEYWORDS}'")
    
    # 处理每篇论文
    papers_with_scores = []
    for paper in papers:
        # 下载论文
        pdf_path = download_paper(paper, PAPERS_DIR)
        if pdf_path:
            # 休眠以避免达到API速率限制
            time.sleep(2)
            
            # 分析论文，传入领域关键词
            analysis = analyze_paper_with_deepseek(pdf_path, paper, DOMAIN_KEYWORDS)
            
            # 提取相关性评分
            score = extract_relevance_score(analysis)
            
            # 存储论文、分析结果和评分
            papers_with_scores.append((paper, analysis, score))
            
            # 分析完成后删除PDF文件
            delete_pdf(pdf_path)
    
    # 按相关性评分降序排序
    papers_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    # 只保留top-N最相关的论文
    top_papers = papers_with_scores[:TOP_PAPERS]
    
    # 将分析结果转换为原有格式（用于兼容现有函数）
    papers_analyses = [(paper, analysis) for paper, analysis, score in top_papers]
    
    # 将分析结果写入conclusion.md（包含所有历史记录）
    write_to_conclusion(top_papers)
    
    # 发送邮件（只包含当天分析的论文）
    email_content = format_email_content(top_papers)
    send_email(email_content)
    
    logger.info(f"ArXiv论文追踪和分析完成，已处理{len(papers)}篇论文，选出{len(top_papers)}篇最相关论文")
    logger.info(f"结果已保存至 {CONCLUSION_FILE.absolute()}")

if __name__ == "__main__":
    main()
