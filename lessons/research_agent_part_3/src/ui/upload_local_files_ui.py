"""Upload UI handlers for local file uploads."""

import logging
import uuid

from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from ..app.notebook_handler import NotebookToMarkdownConverter
from ..db.models import Article, LocalFile
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)

# HTML template for the local files upload page
UPLOAD_LOCAL_FILES_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Local Files - Nova Research</title>
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
            padding: 20px;
        }
        .container {
            background: #ffffff;
            border-radius: 16px;
            padding: 40px;
            max-width: 700px;
            margin: 0 auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        h1 {
            color: #1a1a2e;
            font-size: 24px;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
            margin-bottom: 16px;
        }
        .info-box {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 32px;
            font-size: 13px;
            color: #0c4a6e;
        }
        .progress-bar {
            background: #f0f0f0;
            border-radius: 8px;
            height: 8px;
            margin-bottom: 24px;
            overflow: hidden;
        }
        .progress-fill {
            background: linear-gradient(90deg, #4a90d9, #22c55e);
            height: 100%;
            transition: width 0.3s ease;
        }
        .progress-text {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 32px;
        }
        .file-list {
            margin-bottom: 32px;
        }
        .file-list-title {
            color: #333;
            font-weight: 600;
            margin-bottom: 16px;
            font-size: 16px;
        }
        .file-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 8px;
            transition: all 0.2s;
        }
        .file-item.uploaded {
            background: #f0fdf4;
            border: 1px solid #22c55e;
        }
        .file-status {
            width: 24px;
            height: 24px;
            margin-right: 12px;
            font-size: 18px;
        }
        .file-name {
            flex: 1;
            color: #333;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
        }
        .file-item.uploaded .file-name {
            color: #166534;
        }
        .upload-section {
            border-top: 2px solid #e0e0e0;
            padding-top: 32px;
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
        .selected-file {
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
        .message {
            margin-top: 16px;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            display: none;
        }
        .message.success {
            display: block;
            background: #f0fdf4;
            border: 1px solid #22c55e;
            color: #166534;
        }
        .message.error {
            display: block;
            background: #fef2f2;
            border: 1px solid #ef4444;
            color: #dc2626;
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
        .completion-message {
            text-align: center;
            padding: 40px 20px;
            display: none;
        }
        .completion-message.show {
            display: block;
        }
        .completion-icon {
            font-size: 64px;
            margin-bottom: 16px;
        }
        .completion-message h2 {
            color: #22c55e;
            margin-bottom: 12px;
        }
        .completion-message p {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container" id="mainContainer">
        <!-- Content will be injected by JavaScript -->
    </div>
    
    <script>
        // Get query params
        const urlParams = new URLSearchParams(window.location.search);
        const userId = urlParams.get('user_id');
        const articleGuidelineId = urlParams.get('article_guideline_id');
        const mainContainer = document.getElementById('mainContainer');
        
        let expectedFiles = [];
        let uploadedFiles = new Set();
        
        if (!userId || !articleGuidelineId) {
            // Show error if required params are missing
            mainContainer.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h2>Missing Parameters</h2>
                    <p>This page requires user_id and article_guideline_id parameters. Please access it through the Nova Research agent.</p>
                </div>
            `;
        } else {
            // Fetch expected files and uploaded files
            loadFilesData();
        }
        
        async function loadFilesData() {
            try {
                const encodedUserId = encodeURIComponent(userId);
                const encodedGuidelineId = encodeURIComponent(articleGuidelineId);
                const url = `/upload_local_files/status?user_id=${encodedUserId}&article_guideline_id=${encodedGuidelineId}`;
                const response = await fetch(url);
                const data = await response.json();
                
                if (!response.ok) {
                    showError(data.error || 'Failed to load file data');
                    return;
                }
                
                expectedFiles = data.expected_files || [];
                uploadedFiles = new Set(data.uploaded_files || []);
                
                renderPage();
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        }
        
        function renderPage() {
            const uploadedCount = uploadedFiles.size;
            const suggestedCount = expectedFiles.length;
            
            // Build the suggested files section
            let suggestedFilesSection = '';
            if (suggestedCount > 0) {
                const matchedCount = expectedFiles.filter(file => uploadedFiles.has(normalizeFilename(file))).length;
                const progressPercent = (matchedCount / suggestedCount) * 100;
                
                suggestedFilesSection = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPercent}%"></div>
                    </div>
                    <div class="progress-text">${matchedCount} of ${suggestedCount} suggested files uploaded</div>
                    
                    <div class="file-list">
                        <div class="file-list-title">Suggested Files (from article guideline):</div>
                        ${expectedFiles.map(file => `
                            <div class="file-item ${uploadedFiles.has(normalizeFilename(file)) ? 'uploaded' : ''}">
                                <span class="file-status">${uploadedFiles.has(normalizeFilename(file)) ? '✅' : '💡'}</span>
                                <span class="file-name">${file}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            const infoMessage = suggestedCount > 0
                ? 'The files below were referenced in your article guideline and are shown as suggestions.'
                : 'No files were referenced in your article guideline, but you can still upload any files you need.';
            mainContainer.innerHTML = `
                <h1>Upload Local Files</h1>
                <p class="subtitle">Upload any local files you want to include in your research</p>
                <div class="info-box">
                    ℹ️ You can upload any files you need. ${infoMessage}
                </div>
                
                ${suggestedFilesSection}
                
                ${uploadedCount > 0 ? `
                    <div class="file-list" style="margin-top: ${suggestedCount > 0 ? '24px' : '0'};">
                        <div class="file-list-title">Your Uploaded Files:</div>
                        ${Array.from(uploadedFiles).map(file => `
                            <div class="file-item uploaded">
                                <span class="file-status">✅</span>
                                <span class="file-name">${file}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="upload-section">
                    <form id="uploadForm">
                        <div class="form-group">
                            <label>Select File to Upload</label>
                            <div class="file-input-wrapper" id="fileWrapper">
                                <div class="file-icon">📄</div>
                                <div class="file-text">Click or drag to upload a file</div>
                                <div class="selected-file" id="selectedFile"></div>
                                <input type="file" id="fileInput" name="file" required>
                            </div>
                        </div>
                        
                        <button type="submit" id="submitBtn">Upload File</button>
                    </form>
                    
                    <div class="message" id="message"></div>
                </div>
            `;
            
            setupFormHandlers();
        }
        
        function setupFormHandlers() {
            const form = document.getElementById('uploadForm');
            const fileInput = document.getElementById('fileInput');
            const fileWrapper = document.getElementById('fileWrapper');
            const selectedFile = document.getElementById('selectedFile');
            const submitBtn = document.getElementById('submitBtn');
            const message = document.getElementById('message');
            
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    selectedFile.textContent = this.files[0].name;
                    fileWrapper.classList.add('has-file');
                } else {
                    selectedFile.textContent = '';
                    fileWrapper.classList.remove('has-file');
                }
            });
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const file = fileInput.files[0];
                if (!file) {
                    showMessage('Please select a file to upload', 'error');
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.textContent = 'Uploading...';
                
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const encodedUserId = encodeURIComponent(userId);
                    const encodedGuidelineId = encodeURIComponent(articleGuidelineId);
                    const uploadUrl = `/upload_local_files?user_id=${encodedUserId}&article_guideline_id=${encodedGuidelineId}`;
                    const response = await fetch(uploadUrl, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showMessage(data.message, 'success');
                        // Reload the page data after a short delay
                        setTimeout(() => {
                            loadFilesData();
                        }, 500);
                    } else {
                        showMessage(data.error || 'Upload failed', 'error');
                    }
                } catch (error) {
                    showMessage('Network error: ' + error.message, 'error');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Upload File';
                }
            });
        }
        
        function showMessage(text, type) {
            const message = document.getElementById('message');
            message.textContent = text;
            message.className = `message ${type}`;
        }
        
        function showError(text) {
            mainContainer.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h2>Error</h2>
                    <p>${text}</p>
                </div>
            `;
        }
        
        function normalizeFilename(filename) {
            // Replace path separators with underscores
            return filename.replace(/[\\/]/g, '_');
        }
    </script>
</body>
</html>
"""


async def get_local_files_upload_page(request: Request) -> HTMLResponse:
    """
    Serve the local files upload page.

    The page reads user_id and article_guideline_id from query params via JavaScript.
    If either is missing, it shows an error message.

    Args:
        request: The incoming Starlette request

    Returns:
        HTMLResponse with the upload form
    """
    return HTMLResponse(content=UPLOAD_LOCAL_FILES_PAGE_HTML)


async def get_upload_status(request: Request) -> JSONResponse:
    """
    Get the upload status: expected files and uploaded files.

    Expects:
        - user_id: Query parameter with Descope user ID
        - article_guideline_id: Query parameter with article UUID

    Returns:
        JSONResponse with expected_files and uploaded_files lists
    """
    try:
        # Get params from query
        user_id = request.query_params.get("user_id")
        article_guideline_id = request.query_params.get("article_guideline_id")

        if not user_id or not article_guideline_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing user_id or article_guideline_id query parameter"},
            )

        # Convert to UUID
        try:
            article_uuid = uuid.UUID(article_guideline_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid article_guideline_id format"},
            )

        session_factory = await get_async_session_factory()
        async with session_factory() as session:
            # Get article
            article = await session.get(Article, article_uuid)

            if not article:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Article with ID '{article_guideline_id}' not found"},
                )

            # Verify user ownership
            if article.user_id != user_id:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied"},
                )

            # Get expected files from extracted_urls
            expected_files = []
            if article.extracted_urls and "local_file_paths" in article.extracted_urls:
                expected_files = article.extracted_urls["local_file_paths"]

            # Get uploaded files
            stmt = select(LocalFile).where(
                LocalFile.article_guideline_id == article_uuid,
                LocalFile.user_id == user_id,
            )
            result = await session.execute(stmt)
            uploaded_files_objs = result.scalars().all()
            uploaded_files = [f.filename for f in uploaded_files_objs]

            return JSONResponse(
                status_code=200,
                content={
                    "expected_files": expected_files,
                    "uploaded_files": uploaded_files,
                },
            )

    except Exception as e:
        logger.exception("Error getting upload status")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
        )


async def post_upload_local_file(request: Request) -> JSONResponse:
    """
    Handle local file upload.

    Expects:
        - user_id: Query parameter with Descope user ID
        - article_guideline_id: Query parameter with article UUID
        - file: Multipart form file upload

    Returns:
        JSONResponse with success message and upload status
    """
    try:
        # Get params from query
        user_id = request.query_params.get("user_id")
        article_guideline_id = request.query_params.get("article_guideline_id")

        if not user_id or not article_guideline_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing user_id or article_guideline_id query parameter"},
            )

        # Convert to UUID
        try:
            article_uuid = uuid.UUID(article_guideline_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid article_guideline_id format"},
            )

        # Parse multipart form data
        form = await request.form()
        file = form.get("file")

        if not file:
            return JSONResponse(
                status_code=400,
                content={"error": "No file uploaded"},
            )

        # Get filename
        filename = getattr(file, "filename", "unknown")

        # Normalize filename (replace path separators with underscores)
        normalized_filename = filename.replace("/", "_").replace("\\", "_")

        # Read file content
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        if not content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "File is empty"},
            )

        # Handle .ipynb files specially
        if filename.lower().endswith(".ipynb"):
            try:
                # Convert notebook to markdown
                import tempfile
                from pathlib import Path

                # Write to temp file for conversion
                with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)

                try:
                    notebook_converter = NotebookToMarkdownConverter(include_outputs=True, include_metadata=False)
                    content = notebook_converter.convert_notebook_to_string(tmp_path)
                    # Update filename to .md
                    normalized_filename = normalized_filename.rsplit(".ipynb", 1)[0] + ".md"
                finally:
                    # Clean up temp file
                    tmp_path.unlink()

            except Exception as e:
                logger.error(f"Error converting notebook: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to convert notebook: {str(e)}"},
                )

        session_factory = await get_async_session_factory()
        async with session_factory() as session:
            # Verify article exists and belongs to user
            article = await session.get(Article, article_uuid)

            if not article:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Article with ID '{article_guideline_id}' not found"},
                )

            if article.user_id != user_id:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied"},
                )

            # Check if file already exists
            stmt = select(LocalFile).where(
                LocalFile.article_guideline_id == article_uuid,
                LocalFile.user_id == user_id,
                LocalFile.filename == normalized_filename,
            )
            result = await session.execute(stmt)
            existing_file = result.scalar_one_or_none()

            if existing_file:
                # Update existing file
                existing_file.content = content
                message = f"File '{normalized_filename}' updated successfully"
            else:
                # Create new file
                local_file = LocalFile(
                    user_id=user_id,
                    article_guideline_id=article_uuid,
                    filename=normalized_filename,
                    content=content,
                )
                session.add(local_file)
                message = f"File '{normalized_filename}' uploaded successfully"

            await session.commit()

        logger.info(f"Uploaded local file {normalized_filename} for article {article_guideline_id}")

        return JSONResponse(
            status_code=201,
            content={
                "message": message,
                "filename": normalized_filename,
            },
        )

    except Exception as e:
        logger.exception("Error uploading local file")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
        )
