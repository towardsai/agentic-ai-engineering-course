"""Upload UI handlers for article guideline file uploads."""

import logging
import uuid

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from ..db.models import Article, ArticleStatus
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)

# HTML template for the upload page
UPLOAD_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Article Guideline - Nova Research</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        h1 {
            color: #1a1a2e;
            font-size: 24px;
            margin-bottom: 8px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
            text-align: center;
            margin-bottom: 32px;
        }
        .form-group {
            margin-bottom: 24px;
        }
        label {
            display: block;
            color: #333;
            font-weight: 500;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .file-input-wrapper {
            position: relative;
            border: 2px dashed #e0e0e0;
            border-radius: 8px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-input-wrapper:hover {
            border-color: #4a90d9;
            background: #f8fafc;
        }
        .file-input-wrapper.has-file {
            border-color: #22c55e;
            background: #f0fdf4;
        }
        .file-input-wrapper input[type="file"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        .file-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }
        .file-text {
            color: #666;
            font-size: 14px;
        }
        .file-name {
            color: #22c55e;
            font-weight: 500;
            margin-top: 8px;
        }
        button {
            width: 100%;
            padding: 14px 24px;
            background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 144, 217, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .result {
            margin-top: 24px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        .result.success {
            display: block;
            background: #f0fdf4;
            border: 1px solid #22c55e;
        }
        .result.error {
            display: block;
            background: #fef2f2;
            border: 1px solid #ef4444;
        }
        .result-title {
            font-weight: 600;
            margin-bottom: 12px;
        }
        .result.success .result-title {
            color: #166534;
        }
        .result.error .result-title {
            color: #dc2626;
        }
        .article-id {
            background: #1a1a2e;
            color: #22c55e;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            word-break: break-all;
        }
        .error-message {
            color: #dc2626;
            font-size: 14px;
        }
        .error-container {
            text-align: center;
            padding: 40px 20px;
        }
        .error-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        .error-container h2 {
            color: #dc2626;
            margin-bottom: 12px;
        }
        .error-container p {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container" id="mainContainer">
        <!-- Content will be injected by JavaScript based on user_id presence -->
    </div>
    
    <script>
        // Get user_id from URL query params
        const urlParams = new URLSearchParams(window.location.search);
        const userId = urlParams.get('user_id');
        const mainContainer = document.getElementById('mainContainer');
        
        if (!userId) {
            // Show error if user_id is missing
            mainContainer.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h2>Missing User ID</h2>
                    <p>This page requires a user ID. Please access it through the Nova Research agent, which will
                    provide the correct URL with your user ID.</p>
                </div>
            `;
        } else {
            // Show upload form
            mainContainer.innerHTML = `
                <h1>Upload Article Guideline</h1>
                <p class="subtitle">Upload your article guideline file to get started with Nova Research</p>
                
                <form id="uploadForm">
                    <div class="form-group">
                        <label>Article Guideline File (.md)</label>
                        <div class="file-input-wrapper" id="fileWrapper">
                            <div class="file-icon">📄</div>
                            <div class="file-text">Click or drag to upload your markdown file</div>
                            <div class="file-name" id="fileName"></div>
                            <input type="file" id="fileInput" name="file" accept=".md" required>
                        </div>
                    </div>
                    
                    <button type="submit" id="submitBtn">Upload Article Guideline</button>
                </form>
                
                <div class="result" id="result">
                    <div class="result-title" id="resultTitle"></div>
                    <div id="resultContent"></div>
                </div>
            `;
            
            // Set up form handling
            const form = document.getElementById('uploadForm');
            const fileInput = document.getElementById('fileInput');
            const fileWrapper = document.getElementById('fileWrapper');
            const fileName = document.getElementById('fileName');
            const submitBtn = document.getElementById('submitBtn');
            const result = document.getElementById('result');
            const resultTitle = document.getElementById('resultTitle');
            const resultContent = document.getElementById('resultContent');
            
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    fileName.textContent = this.files[0].name;
                    fileWrapper.classList.add('has-file');
                } else {
                    fileName.textContent = '';
                    fileWrapper.classList.remove('has-file');
                }
            });
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const file = fileInput.files[0];
                
                if (!file) {
                    showError('Please select a file to upload');
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.textContent = 'Uploading...';
                
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const response = await fetch(`/upload_article_guideline?user_id=${encodeURIComponent(userId)}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showSuccess(data.article_id);
                    } else {
                        showError(data.error || 'Upload failed');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Upload Article Guideline';
                }
            });
            
            function showSuccess(articleId) {
                result.className = 'result success';
                resultTitle.textContent = 'Upload Successful!';
                resultContent.innerHTML = `
                    <p style="margin-bottom: 12px; color: #166534;">Your article guideline ID:</p>
                    <div class="article-id">
                        <span id="articleIdText">${articleId}</span>
                    </div>
                    <p style="margin-top: 12px; color: #666; font-size: 13px;">
                        Copy this ID and provide it to the Nova Research agent to continue.
                    </p>
                `;
            }
            
            function showError(message) {
                result.className = 'result error';
                resultTitle.textContent = 'Error';
                resultContent.innerHTML = `<p class="error-message">${message}</p>`;
            }
        }
    </script>
</body>
</html>
"""


async def get_upload_page(request: Request) -> HTMLResponse:
    """
    Serve the article guideline upload page.

    The page reads user_id from query params via JavaScript.
    If user_id is missing, it shows an error message.

    Args:
        request: The incoming Starlette request

    Returns:
        HTMLResponse with the upload form
    """
    return HTMLResponse(content=UPLOAD_PAGE_HTML)


async def post_upload_guideline(request: Request) -> JSONResponse:
    """
    Handle article guideline file upload.

    Expects:
        - user_id: Query parameter with Descope user ID
        - file: Multipart form file upload (.md file)

    Returns:
        JSONResponse with article_id on success, or error message on failure
    """
    try:
        # Get user_id from query params
        user_id = request.query_params.get("user_id")
        if not user_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing user_id query parameter"},
            )

        # Parse multipart form data
        form = await request.form()
        file = form.get("file")

        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"},
            )

        # Read file content
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        if not content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "File is empty"},
            )

        # Generate article ID
        article_id = uuid.uuid4()

        # Save to database
        session_factory = await get_async_session_factory()
        async with session_factory() as session:
            article = Article(
                id=article_id,
                user_id=user_id,
                guideline_text=content,
                status=ArticleStatus.CREATED,
            )
            session.add(article)
            await session.commit()

        logger.info(f"Created article {article_id} for user {user_id}")

        return JSONResponse(
            status_code=201,
            content={
                "article_id": str(article_id),
                "message": "Article guideline uploaded successfully",
            },
        )

    except Exception as e:
        logger.exception("Error uploading article guideline")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
        )
