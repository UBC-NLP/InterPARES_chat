import re
import logging
import json
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import requests
from datetime import datetime
from uuid import uuid4
import random
import configparser
def getconfig(configfile_path:str):
    """
    Read the config file

    Params
    ----------------
    configfile_path: file path of .cfg file
    """

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configfile_path))
        return config
    except:
        logging.warning("config file not found")
        return None
        

def save_logs(json_path, logs, feedback=None):
    """Save logs to local JSON file only (no HuggingFace upload)"""
    if logs is None:
        logging.error("save_logs received None for logs parameter")
        return
    
    try:
        from datetime import datetime
        current_time = datetime.now()
        logs["time"] = str(current_time)
        
        if feedback is not None:
            logs["feedback"] = feedback
        
        # Save to local JSON file
        with open(json_path, 'a') as f:
            f.write(json.dumps(logs) + '\n')
            
        logging.info(f"Logs saved successfully to {json_path}")
    except Exception as e:
        logging.error(f"Error in save_logs: {e}")
        raise

def get_message_template(type, SYSTEM_PROMPT, USER_PROMPT):
    if type == 'OTHERS':
        messages =  [{"role": "system", "content": SYSTEM_PROMPT},
                {"role":"user","content":USER_PROMPT}]
    elif type == 'DEDICATED':
        messages = [
                 SystemMessage(content=SYSTEM_PROMPT),
                 HumanMessage(content=USER_PROMPT),]
    elif type == 'SERVERLESS':
        messages = [
                 SystemMessage(content=SYSTEM_PROMPT),
                 HumanMessage(content=USER_PROMPT),]
    elif type == 'INF_PROVIDERS':
        messages =  [{"role": "system", "content": SYSTEM_PROMPT},
                {"role":"user","content":USER_PROMPT}]
    else:
        logging.error("No message template found")
        raise
    
    return messages


def make_html_source(source,i):
    """
    takes the text and converts it into displayable content using fitz for PDF processing
    """
    import os
    import base64
    import io
    import urllib.parse
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        return make_html_source_fallback(source, i)
    
    base_path = "."
    meta = source.metadata
    content = source.page_content.strip()

    name = meta['filename_org']
    full_pdf_path = os.path.join(base_path, name)
    # Use custom download endpoint with URL encoding
    pdf_url = f"/download_pdf/{urllib.parse.quote(name)}"
    page_num = int(meta['page']) - 1  # fitz uses 0-based indexing
    
    print(f"Processing PDF with fitz: {full_pdf_path}, page {page_num + 1}")
    
    # Check if file exists
    if not os.path.exists(full_pdf_path):
        print(f"Warning: PDF file not found at {full_pdf_path}")
        return make_html_source_fallback(source, i)
    
    try:
        # Open PDF with fitz
        doc = fitz.open(full_pdf_path)
        
        # Check if page exists
        if page_num >= len(doc) or page_num < 0:
            print(f"Page {page_num + 1} not found in PDF")
            doc.close()
            return make_html_source_fallback(source, i)
        
        # Get the page
        page = doc[page_num]
        
        # Convert page to image (PNG)
        mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        pix = None  # Free memory
        
        # Convert to base64 for embedding
        img_base64 = base64.b64encode(img_data).decode()
        
        # Close document
        doc.close()
        
        # Create HTML with embedded image
        card = f"""
            <div class="card" id="doc{i}">
                <div class="card-content">
                    <h2 style="font-size: 12px;"><a href="{pdf_url}" style="text-decoration: none; color: inherit;" target="_blank">Doc {i} - {os.path.basename(name)} - Page {int(meta['page'])}</a></h2>
                    <div class="pdf-display">
                        <div class="pdf-image-container" style="text-align: center; margin: 10px 0;">
                            <img src="data:image/png;base64,{img_base64}" 
                                 style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px;"
                                 alt="PDF Page {int(meta['page'])}" />
                        </div>
                        <div class="content-preview" style="margin-top: 15px;">
                            <h4>Extracted Content:</h4>
                            <div class="content-text" style="background: #f8f9fa; padding: 12px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-size: 14px; line-height: 1.4;">
                                <p>{content}</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card-footer" style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background: #f1f3f4;">
                    <span style="font-size: 12px; color: #666;">{os.path.basename(name)}</span>
                    <span style="font-size: 12px; color: #666;">Page {int(meta['page'])}</span>
                </div>
            </div>
            """
        
        return card
        
    except Exception as e:
        print(f"Error processing PDF with fitz: {e}")
        return make_html_source_fallback(source, i)

def make_html_source_fallback(source, i):
    """
    Fallback function when fitz fails or PDF is not accessible
    """
    import os
    import urllib.parse
    meta = source.metadata
    content = source.page_content.strip()
    name = meta['filename_org']
    
    # Construct full path for download link
    base_path = "."
    full_pdf_path = os.path.join(base_path, name)
    # Use custom download endpoint with URL encoding
    pdf_url = f"/download_pdf/{urllib.parse.quote(name)}"
    
    card = f"""
        <div class="card" id="doc{i}">
            <div class="card-content">
                <h2 style="font-size: 12px;"><a href="{pdf_url}" style="text-decoration: none; color: inherit;" target="_blank">Doc {i} - {os.path.basename(name)} - Page {int(meta['page'])}</a></h2>
                <div class="content-display">
                    <div class="pdf-unavailable" style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <p style="margin: 0; color: #856404;"><strong>📄 PDF Preview Unavailable</strong></p>
                        <p style="margin: 5px 0 0 0; color: #856404; font-size: 12px;">Showing extracted text content instead</p>
                    </div>
                    <div class="content-preview">
                        <h4>Content:</h4>
                        <div class="content-text" style="background: #f8f9fa; padding: 12px; border-radius: 5px; max-height: 400px; overflow-y: auto; font-size: 14px; line-height: 1.4;">
                            <p>{content}</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-footer" style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background: #f1f3f4;">
                <span style="font-size: 12px; color: #666;">{os.path.basename(name)}</span>
                <span style="font-size: 12px; color: #666;">Page {int(meta['page'])}</span>
            </div>
        </div>
        """
    
    return card


def parse_output_llm_with_sources(output):
    # Split the content into a list of text and "[Doc X]" references
    content_parts = re.split(r'\[(Doc\s?\d+(?:,\s?Doc\s?\d+)*)\]', output)
    parts = []
    for part in content_parts:
        if part.startswith("Doc"):
            subparts = part.split(",")
            subparts = [subpart.lower().replace("doc","").strip() for subpart in subparts]
            subparts = [f"""<a href="#doc{subpart}" class="a-doc-ref" target="_self"><span class='doc-ref'><sup>{subpart}</sup></span></a>""" for subpart in subparts]
            parts.append("".join(subparts))
        else:
            parts.append(part)
    content_parts = "".join(parts)
    return content_parts


def get_client_ip(request=None):
    """Get the client IP address from the request context"""
    try:
        if request:
            # Try different headers that might contain the real IP
            ip = request.client.host
            # Check for proxy headers
            forwarded_for = request.headers.get('X-Forwarded-For')
            if forwarded_for:
                # X-Forwarded-For can contain multiple IPs - first one is the client
                ip = forwarded_for.split(',')[0].strip()
            
            logging.debug(f"Client IP detected: {ip}")
            return ip
    except Exception as e:
        logging.error(f"Error getting client IP: {e}")
    return "127.0.0.1"


def get_client_location(ip_address) -> dict | None:
    """Get geolocation info using ipapi.co"""
    # Add headers so we don't get blocked...
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(
            f'https://ipapi.co/{ip_address}/json/',
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Add random noise between -0.01 and 0.01 degrees (roughly ±1km)
            lat = data.get('latitude')
            lon = data.get('longitude')
            if lat is not None and lon is not None:
                lat += random.uniform(-0.01, 0.01)
                lon += random.uniform(-0.01, 0.01)
            
            return {
                'city': data.get('city'),
                'region': data.get('region'),
                'country': data.get('country_name'),
                'latitude': lat,
                'longitude': lon
            }
        elif response.status_code == 429:
            logging.warning(f"Rate limit exceeded. Response: {response.text}")
            return None
        else:
            logging.error(f"Error: Status code {response.status_code}. Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return None


def get_platform_info(user_agent: str) -> str:
    """Get platform info"""
    # Make a best guess at the device type
    if any(mobile_keyword in user_agent.lower() for mobile_keyword in ['mobile', 'android', 'iphone', 'ipad', 'ipod']):
        platform_info = 'mobile'
    else:
        platform_info = 'desktop'
            
    return platform_info