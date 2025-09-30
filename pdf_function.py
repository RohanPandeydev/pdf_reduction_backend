import os
import boto3
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_file, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import io
import re
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import shutil
import logging
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
import requests
from pymongo.server_api import ServerApi
import dns.resolver
from flask_cors import CORS

# Configure DNS resolver to use Google's DNS
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']




# Run Command
# py -3.12 --version
# py -3.12 -m venv venv
# source venv/Scripts/activate
# pip install -r requirements.txt


load_dotenv()  # loads variables from .env

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
CORS(app)

# CORS Configuration - Updated for Netlify frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://pdf-reduction.netlify.app",
            "http://localhost:3000",  # For local development
            "http://localhost:5173",  # For Vite local development
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    },
    r"/upload-pdf": {
        "origins": [
            "https://pdf-reduction.netlify.app",
            "http://localhost:3000",
            "http://localhost:5173",
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    },
    r"/health": {
        "origins": "*",  # Public endpoint
        "methods": ["GET", "OPTIONS"]
    }
})

# Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'eu-north-1')
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

# MongoDB Configuration
# MONGO_URI = "mongodb://127.0.0.1:27017"  # localhost
MONGO_URI = "mongodb+srv://rohankrpandey20_db_user:rohan123@cluster0.nkv1lfk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'pdf_redaction_db')

# Debug AWS configuration
logger.info("=== AWS Configuration Debug ===")
logger.info(f"AWS_ACCESS_KEY_ID exists: {AWS_ACCESS_KEY is not None}")
logger.info(f"AWS_SECRET_ACCESS_KEY exists: {AWS_SECRET_KEY is not None}")
logger.info(f"AWS_REGION: {AWS_REGION}")
logger.info(f"S3_BUCKET_NAME: {S3_BUCKET}")

# Local storage configuration
LOCAL_STORAGE_PATH = os.path.join(os.getcwd(), 'public', 'uploads')
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)
logger.info(f"Local storage path: {LOCAL_STORAGE_PATH}")

# Initialize MongoDB client
mongo_client = None
mongo_db = None
mongo_available = False

# Now create MongoDB connection
try:
    client = MongoClient(
        MONGO_URI,
        server_api=ServerApi('1'),
        serverSelectionTimeoutMS=10000
    )
    client.admin.command('ping')
    print("✓ Successfully connected to MongoDB!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    
def initialize_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, mongo_db, mongo_available
    
    try:
        logger.info("Connecting to MongoDB...")
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        mongo_client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        mongo_db = mongo_client[MONGO_DB_NAME]
        
        # Create indexes for efficient searching
        pdfs_collection = mongo_db['pdfs']
        pdfs_collection.create_index([('filename', ASCENDING)])
        pdfs_collection.create_index([('upload_timestamp', DESCENDING)])
        pdfs_collection.create_index([('original_filename', 'text'), ('filename', 'text')])
        
        logger.info("MongoDB indexes created successfully")
        mongo_available = True
        return True
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection failed: {e}")
        mongo_available = False
        return False
    except Exception as e:
        logger.error(f"Unexpected MongoDB error: {e}")
        mongo_available = False
        return False


# Initialize MongoDB
initialize_mongodb()


# Initialize S3 client
s3_client = None
s3_initialization_error = None
s3_available = False

def initialize_s3_client():
    """Initialize S3 client with comprehensive error handling"""
    global s3_client, s3_initialization_error, s3_available
    
    if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET]):
        missing = []
        if not AWS_ACCESS_KEY:
            missing.append("AWS_ACCESS_KEY_ID")
        if not AWS_SECRET_KEY:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if not S3_BUCKET:
            missing.append("S3_BUCKET_NAME")
        
        s3_initialization_error = f"Missing environment variables: {', '.join(missing)}"
        logger.error(s3_initialization_error)
        return False
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        logger.info("Testing S3 connection...")
        response = s3_client.list_buckets()
        logger.info(f"Successfully connected to S3. Found {len(response['Buckets'])} buckets")
        
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
            logger.info(f"Bucket '{S3_BUCKET}' exists and is accessible")
            s3_available = True
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                s3_initialization_error = f"Bucket '{S3_BUCKET}' does not exist"
            elif error_code == '403':
                s3_initialization_error = f"Access denied to bucket '{S3_BUCKET}'"
            else:
                s3_initialization_error = f"Error accessing bucket: {e}"
            logger.error(s3_initialization_error)
            return False
            
    except Exception as e:
        s3_initialization_error = f"S3 initialization error: {e}"
        logger.error(s3_initialization_error)
        return False

# Initialize S3
initialize_s3_client()

class PDFHandler:
    def __init__(self):
        self.s3_client = s3_client
        self.bucket_name = S3_BUCKET
        self.local_path = LOCAL_STORAGE_PATH
        self.s3_available = s3_available
        self.db = mongo_db
        self.aws_region = AWS_REGION

    def get_s3_url_from_key(self, s3_key):
        """Generate full S3 URL from S3 key"""
        return f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"

    def fetch_pdf_from_s3(self, s3_key):
        """Fetch PDF content from S3"""
        try:
            logger.info(f"Fetching PDF from S3: {s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            pdf_content = response['Body'].read()
            
            logger.info(f"Successfully fetched PDF from S3. Size: {len(pdf_content)} bytes")
            return pdf_content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"S3 fetch error ({error_code}): {e}")
            raise Exception(f"Failed to fetch PDF from S3: {error_code}")
        except Exception as e:
            logger.error(f"Unexpected error fetching from S3: {e}")
            raise Exception(f"Failed to fetch PDF: {str(e)}")

    def get_pdf_by_id(self, pdf_id):
        """Fetch PDF metadata from MongoDB by ID"""
        try:
            if not mongo_available:
                raise Exception("Database not available")
            
            pdfs_collection = self.db['pdfs']
            pdf = pdfs_collection.find_one({'_id': ObjectId(pdf_id), 'status': 'active'})
            
            if not pdf:
                raise Exception(f"PDF with ID {pdf_id} not found")
            
            return pdf
            
        except Exception as e:
            logger.error(f"Error fetching PDF metadata: {e}")
            raise

    def save_pdf_metadata(self, filename, original_filename, s3_key, file_size, is_redacted=False, parent_id=None):
        """Save PDF metadata to MongoDB"""
        if not mongo_available or self.db is None:
            logger.warning("MongoDB not available, skipping metadata save")
            return None

        try:
            pdfs_collection = self.db['pdfs']

            metadata = {
                'filename': filename,
                'original_filename': original_filename,
                's3_key': s3_key,
                'storage_type': 's3' if s3_key.startswith('pdfs/') else 'local',
                'mime_type': 'application/pdf',
                'file_size': file_size,
                'upload_timestamp': datetime.utcnow(),
                'is_redacted': is_redacted,
                'parent_id': parent_id,
                'status': 'active'
            }

            result = pdfs_collection.insert_one(metadata)
            logger.info(f"PDF metadata saved with ID: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error saving PDF metadata: {e}")
            return None

    def upload_pdf_to_s3(self, file, filename, file_size):
        """Upload PDF file to S3 with detailed error handling"""
        logger.info(f"Attempting to upload file: {filename}")

        # Check S3 availability first
        if not self.s3_available:
            error_msg = f"S3 not available: {s3_initialization_error or 'Unknown error'}"
            logger.error(error_msg)
            raise Exception(error_msg)

        if not self.s3_client:
            raise Exception("S3 client not initialized")

        try:
            s3_key = f"pdfs/{filename}"
            logger.info(f"Uploading to S3 bucket: {self.bucket_name}, key: {s3_key}")

            file.seek(0)

            self.s3_client.upload_fileobj(
                file,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'application/pdf',
                    'ServerSideEncryption': 'AES256'
                }
            )

            logger.info(f"Successfully uploaded to S3: {s3_key}")

            # Verify upload
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                logger.info("Upload verified successfully")
            except ClientError as e:
                logger.warning(f"Upload verification failed: {e}")

            return s3_key

        except NoCredentialsError:
            error_msg = "AWS credentials not found or invalid"
            logger.error(error_msg)
            raise Exception(error_msg)
        except PartialCredentialsError:
            error_msg = "Incomplete AWS credentials"
            logger.error(error_msg)
            raise Exception(error_msg)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"AWS S3 error ({error_code}): {e.response['Error']['Message']}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Unexpected S3 upload error: {str(e)}")
            raise Exception(f"Failed to upload to S3: {str(e)}")

    def save_to_local(self, file, filename):
        """Save file to local storage as fallback"""
        try:
            local_file_path = os.path.join(self.local_path, filename)
            file.seek(0)
            file.save(local_file_path)
            logger.info(f"Saved to local storage: {local_file_path}")
            return f"uploads/{filename}"
        except Exception as e:
            logger.error(f"Failed to save to local storage: {e}")
            raise Exception(f"Failed to save file: {e}")

    def search_word_in_pdf(self, pdf_content, search_word):
        """Search for word in PDF and return positions"""
        logger.info(f"Searching for word: '{search_word}' in PDF")
        results = []
        
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        try:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                
                search_flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE
                text_instances = page.search_for(search_word, flags=search_flags)
                exact_instances = page.search_for(search_word, flags=0)
                
                all_instances = list(text_instances) + list(exact_instances)
                unique_instances = []
                for inst in all_instances:
                    is_duplicate = False
                    for existing in unique_instances:
                        if (abs(inst.x0 - existing.x0) < 2 and 
                            abs(inst.y0 - existing.y0) < 2 and
                            abs(inst.x1 - existing.x1) < 2 and 
                            abs(inst.y1 - existing.y1) < 2):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_instances.append(inst)
                
                for inst in unique_instances:
                    page_text = page.get_text()
                    words = re.split(r'\s+', page_text)
                    word_index = -1
                    for i, word in enumerate(words):
                        if search_word.lower() in word.lower():
                            word_index = i
                            break
                    
                    context_start = max(0, word_index - 5)
                    context_end = min(len(words), word_index + 6)
                    context = ' '.join(words[context_start:context_end]) if word_index >= 0 else "Context not found"
                    
                    expanded_rect = fitz.Rect(
                        inst.x0 - 1,
                        inst.y0 - 1,
                        inst.x1 + 1,
                        inst.y1 + 1
                    )
                    
                    results.append({
                        'page': page_num + 1,
                        'position': {
                            'x0': expanded_rect.x0,
                            'y0': expanded_rect.y0,
                            'x1': expanded_rect.x1,
                            'y1': expanded_rect.y1
                        },
                        'context': context,
                        'found_word': search_word
                    })
        finally:
            pdf_doc.close()
        
        logger.info(f"Found {len(results)} matches for '{search_word}'")
        return results

    def true_redaction_pdf(self, pdf_content, search_results, search_term):
        """Perform true redaction"""
        logger.info(f"Performing true redaction for {len(search_results)} matches")
        
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        try:
            results_by_page = {}
            for result in search_results:
                page_num = result['page'] - 1
                if page_num not in results_by_page:
                    results_by_page[page_num] = []
                results_by_page[page_num].append(result)
            
            for page_num, page_results in results_by_page.items():
                page = pdf_doc[page_num]
                
                redaction_rects = []
                for result in page_results:
                    rect = fitz.Rect(
                        result['position']['x0'],
                        result['position']['y0'],
                        result['position']['x1'],
                        result['position']['y1']
                    )
                    
                    redact_annot = page.add_redact_annot(rect)
                    redact_annot.set_colors(fill=[0, 0, 0])
                    redact_annot.update()
                    redaction_rects.append(rect)
                
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
                
                for rect in redaction_rects:
                    rect_height = rect.height
                    rect_width = rect.width
                    
                    font_size = min(max(6, rect_height * 0.6), 14)
                    text_width_estimate =  font_size * 0.6
                    if text_width_estimate > rect_width:
                        font_size = max(4, rect_width / ( 0.6))
                    
                    text_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1)
                    
                    try:
                        page.insert_textbox(
                            text_rect,
                            # "CONFIDENTIAL",
                            fontsize=font_size,
                            color=(1, 1, 1),
                            align=fitz.TEXT_ALIGN_CENTER,
                            fontname="helvetica-bold"
                        )
                    except:
                        center_x = (rect.x0 + rect.x1) / 2
                        center_y = (rect.y0 + rect.y1) / 2
                        # page.insert_text(
                        #     (center_x - 30, center_y + 3),
                        #     "CONFIDENTIAL",
                        #     fontsize=font_size,
                        #     color=(1, 1, 1),
                        #     fontname="helvetica-bold"
                        # )
            
            redacted_pdf_bytes = pdf_doc.write()
            logger.info(f"Redaction completed. Output size: {len(redacted_pdf_bytes)} bytes")
            return redacted_pdf_bytes
        finally:
            pdf_doc.close()

    def save_redacted_pdf(self, pdf_content, original_filename, parent_id=None):
        """Save redacted PDF to S3 and MongoDB"""
        name_without_ext = os.path.splitext(original_filename)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        redacted_filename = f"{name_without_ext}_redacted_{timestamp}_copy.pdf"
        
        logger.info(f"Saving redacted PDF: {redacted_filename}")
        
        if not self.s3_available:
            raise Exception("S3 storage is required for saving redacted PDFs")
        
        try:
            redacted_s3_key = f"pdfs/redacted/{redacted_filename}"
            logger.info(f"Uploading redacted PDF to S3: {redacted_s3_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=redacted_s3_key,
                Body=pdf_content,
                ContentType='application/pdf',
                ServerSideEncryption='AES256'
            )
            
            # Save metadata to MongoDB
            file_size = len(pdf_content)
            metadata_id = self.save_pdf_metadata(
                redacted_filename,
                original_filename,
                redacted_s3_key,
                file_size,
                is_redacted=True,
                parent_id=parent_id
            )
            
            logger.info(f"Successfully saved redacted PDF: {redacted_s3_key}")
            return redacted_s3_key, metadata_id
            
        except Exception as e:
            logger.error(f"Failed to save redacted PDF: {str(e)}")
            raise Exception(f"Failed to save redacted PDF: {str(e)}")

# Initialize PDF handler
pdf_handler = PDFHandler()

# Store search results temporarily
search_cache = {}
@app.route("/")
def home():
    return "Hello, Flask is running on EC2!"


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """API endpoint to upload PDF directly to S3"""
    try:
        logger.info("PDF upload request received")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Get file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        # Secure the filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = f"{timestamp}{filename}"
        
        logger.info(f"Uploading file: {unique_filename}, size: {file_size} bytes")
        
        # Upload directly to S3
        s3_key = pdf_handler.upload_pdf_to_s3(file, unique_filename, file_size)
        
        # Save metadata to MongoDB
        metadata_id = pdf_handler.save_pdf_metadata(
            unique_filename,
            filename,
            s3_key,
            file_size,
            is_redacted=False
        )
        
        return jsonify({
            'message': 'PDF uploaded successfully',
            'filename': unique_filename,
            'original_filename': filename,
            'file_path': s3_key,
            's3_url': pdf_handler.get_s3_url_from_key(s3_key),
            'file_size': file_size,
            'metadata_id': metadata_id,
            'upload_time': datetime.now().isoformat(),
            'storage_type': 's3',
            's3_available': pdf_handler.s3_available
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-pdf-by-id', methods=['POST'])
def api_search_pdf_by_id():
    """NEW: Search PDF by fetching from S3 using MongoDB ID"""
    try:
        data = request.get_json()
        
        if not data or 'pdf_id' not in data or 'searchTerm' not in data:
            return jsonify({
                'success': False, 
                'error': 'Missing pdf_id or searchTerm'
            }), 400
        
        pdf_id = data['pdf_id']
        search_term = data['searchTerm'].strip()
        
        if not search_term:
            return jsonify({
                'success': False, 
                'error': 'Search term cannot be empty'
            }), 400
        
        logger.info(f"Searching PDF with ID: {pdf_id} for term: '{search_term}'")
        
        # Fetch PDF metadata from MongoDB
        pdf_metadata = pdf_handler.get_pdf_by_id(pdf_id)
        s3_key = pdf_metadata['s3_key']
        
        # Generate full S3 URL
        s3_url = pdf_handler.get_s3_url_from_key(s3_key)
        logger.info(f"PDF S3 URL: {s3_url}")
        
        # Fetch PDF content from S3
        pdf_content = pdf_handler.fetch_pdf_from_s3(s3_key)
        
        # Search for the word
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        # Format matches for frontend
        formatted_matches = []
        for result in search_results:
            formatted_matches.append({
                'page': result['page'],
                'text': search_term,
                'x': result['position']['x0'],
                'y': result['position']['y0'],
                'width': result['position']['x1'] - result['position']['x0'],
                'height': result['position']['y1'] - result['position']['y0'],
                'context': result['context']
            })
        
        # Store in cache for later redaction
        session_key = f"search_{pdf_id}_{datetime.now().timestamp()}_{hash(search_term)}"
        search_cache[session_key] = {
            'results': search_results,
            'pdf_content': pdf_content,
            'search_term': search_term,
            'pdf_id': pdf_id,
            'pdf_metadata': pdf_metadata
        }
        
        return jsonify({
            'success': True,
            'matches': formatted_matches,
            'totalMatches': len(formatted_matches),
            'sessionKey': session_key,
            'pdf_info': {
                'id': pdf_id,
                'filename': pdf_metadata['filename'],
                'original_filename': pdf_metadata['original_filename'],
                's3_key': s3_key,
                's3_url': s3_url,
                'file_size': pdf_metadata['file_size']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Search by ID error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/save-redacted-by-id', methods=['POST'])
def api_save_redacted_by_id():
    """NEW: Save redacted PDF after searching by ID"""
    try:
        data = request.get_json()
        
        if not data or 'pdf_id' not in data or 'searchTerm' not in data:
            return jsonify({
                'success': False, 
                'error': 'Missing pdf_id or searchTerm'
            }), 400
        
        pdf_id = data['pdf_id']
        search_term = data['searchTerm'].strip()
        
        logger.info(f"Saving redacted PDF for ID: {pdf_id}, term: '{search_term}'")
        
        # Fetch PDF metadata from MongoDB
        pdf_metadata = pdf_handler.get_pdf_by_id(pdf_id)
        s3_key = pdf_metadata['s3_key']
        
        # Fetch PDF content from S3
        pdf_content = pdf_handler.fetch_pdf_from_s3(s3_key)
        
        # Search for the word
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        if not search_results:
            return jsonify({
                'success': False, 
                'error': 'No matches found to redact'
            }), 400
        
        # Perform redaction
        redacted_pdf = pdf_handler.true_redaction_pdf(pdf_content, search_results, search_term)
        
        # Save redacted PDF to S3 with new name and create new MongoDB entry
        saved_s3_key, new_metadata_id = pdf_handler.save_redacted_pdf(
            redacted_pdf,
            pdf_metadata['original_filename'],
            parent_id=pdf_id  # Link to original PDF
        )
        
        return jsonify({
            'success': True,
            'message': f'Successfully redacted {len(search_results)} instances and saved',
            'totalReplacements': len(search_results),
            'original_pdf': {
                'id': pdf_id,
                'filename': pdf_metadata['filename']
            },
            'redacted_pdf': {
                'id': new_metadata_id,
                'filename': os.path.basename(saved_s3_key),
                's3_key': saved_s3_key,
                's3_url': pdf_handler.get_s3_url_from_key(saved_s3_key)
            },
            'storage_type': 's3'
        }), 200
        
    except Exception as e:
        logger.error(f"Save redacted by ID error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/download-redacted-by-id', methods=['POST'])
def api_download_redacted_by_id():
    """NEW: Download redacted PDF after searching by ID"""
    try:
        data = request.get_json()
        
        if not data or 'pdf_id' not in data or 'searchTerm' not in data:
            return jsonify({
                'success': False, 
                'error': 'Missing pdf_id or searchTerm'
            }), 400
        
        pdf_id = data['pdf_id']
        search_term = data['searchTerm'].strip()
        
        logger.info(f"Downloading redacted PDF for ID: {pdf_id}, term: '{search_term}'")
        
        # Fetch PDF metadata from MongoDB
        pdf_metadata = pdf_handler.get_pdf_by_id(pdf_id)
        s3_key = pdf_metadata['s3_key']
        
        # Fetch PDF content from S3
        pdf_content = pdf_handler.fetch_pdf_from_s3(s3_key)
        
        # Search for the word
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        if not search_results:
            return jsonify({
                'error': 'No matches found to redact'
            }), 400
        
        # Perform redaction
        redacted_pdf = pdf_handler.true_redaction_pdf(pdf_content, search_results, search_term)
        
        # Generate redacted filename
        original_name = os.path.splitext(pdf_metadata['original_filename'])[0]
        redacted_filename = f"{original_name}_redacted.pdf"
        
        logger.info(f"Returning redacted PDF: {redacted_filename}")
        
        return send_file(
            io.BytesIO(redacted_pdf),
            as_attachment=True,
            download_name=redacted_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Download redacted by ID error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/search-pdf', methods=['POST'])
def api_search_pdf():
    """Frontend compatible search endpoint"""
    try:
        logger.info("PDF search request received")
        
        if 'pdf' not in request.files or 'searchTerm' not in request.form:
            return jsonify({'success': False, 'error': 'Missing PDF file or search term'}), 400
        
        file = request.files['pdf']
        search_term = request.form['searchTerm'].strip()
        
        if not search_term:
            return jsonify({'success': False, 'error': 'Search term cannot be empty'}), 400
        
        pdf_content = file.read()
        file.seek(0)
        
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        formatted_matches = []
        for result in search_results:
            formatted_matches.append({
                'page': result['page'],
                'text': search_term,
                'x': result['position']['x0'],
                'y': result['position']['y0'],
                'width': result['position']['x1'] - result['position']['x0'],
                'height': result['position']['y1'] - result['position']['y0'],
                'context': result['context']
            })
        
        session_key = f"search_{datetime.now().timestamp()}_{hash(search_term)}"
        search_cache[session_key] = {
            'results': search_results,
            'pdf_content': pdf_content,
            'search_term': search_term,
            'filename': file.filename
        }
        
        return jsonify({
            'success': True,
            'matches': formatted_matches,
            'totalMatches': len(formatted_matches),
            'sessionKey': session_key
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-pdf', methods=['POST'])
def api_download_pdf():
    """Frontend compatible download endpoint with true redaction"""
    try:
        logger.info("PDF download request received")
        
        if 'pdf' not in request.files or 'searchTerm' not in request.form:
            return jsonify({'error': 'Missing PDF file or search term'}), 400
        
        file = request.files['pdf']
        search_term = request.form['searchTerm'].strip()
        
        pdf_content = file.read()
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        if not search_results:
            return jsonify({'error': 'No matches found to redact'}), 400
        
        redacted_pdf = pdf_handler.true_redaction_pdf(pdf_content, search_results, search_term)
        
        original_name = os.path.splitext(file.filename)[0]
        redacted_filename = f"{original_name}_redacted.pdf"
        
        logger.info(f"Returning redacted PDF: {redacted_filename}")
        
        return send_file(
            io.BytesIO(redacted_pdf),
            as_attachment=True,
            download_name=redacted_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/replace-and-save', methods=['POST'])
def api_replace_and_save():
    """Frontend compatible save endpoint with true redaction"""
    try:
        logger.info("PDF replace and save request received")
        
        if 'pdf' not in request.files or 'searchTerm' not in request.form:
            return jsonify({'success': False, 'error': 'Missing PDF file or search term'}), 400
        
        file = request.files['pdf']
        search_term = request.form['searchTerm'].strip()
        parent_id = request.form.get('parentId')  # Optional: ID of original PDF
        
        pdf_content = file.read()
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        if not search_results:
            return jsonify({'success': False, 'error': 'No matches found to redact'}), 400
        
        redacted_pdf = pdf_handler.true_redaction_pdf(pdf_content, search_results, search_term)
        
        saved_path, metadata_id = pdf_handler.save_redacted_pdf(
            redacted_pdf, 
            file.filename,
            parent_id=parent_id
        )
        
        return jsonify({
            'success': True,
            'message': f'Successfully redacted {len(search_results)} instances and saved to server',
            'totalReplacements': len(search_results),
            'savedFile': saved_path,
            'filename': os.path.basename(saved_path),
            'metadata_id': metadata_id,
            'storage_type': 's3',
            's3_available': pdf_handler.s3_available
        }), 200
        
    except Exception as e:
        logger.error(f"Replace and save error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdfs', methods=['GET'])
def list_pdfs():
    """List all PDFs with search and pagination"""
    try:
        if not mongo_available:
            return jsonify({'error': 'Database not available'}), 503
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search_query = request.args.get('search', '').strip()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        is_redacted = request.args.get('is_redacted')
        
        # Build MongoDB query
        query = {'status': 'active'}
        
        # Search by filename (fuzzy search using regex)
        if search_query:
            # Create a regex pattern for flexible matching
            search_pattern = '.*'.join(re.escape(char) for char in search_query)
            query['$or'] = [
                {'filename': {'$regex': search_pattern, '$options': 'i'}},
                {'original_filename': {'$regex': search_pattern, '$options': 'i'}}
            ]
        
        # Date range filter
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query['$gte'] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                date_query['$lte'] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if date_query:
                query['upload_timestamp'] = date_query
        
        # Filter by redacted status
        if is_redacted is not None:
            query['is_redacted'] = is_redacted.lower() == 'true'
        
        pdfs_collection = mongo_db['pdfs']
        
        # Count total documents
        total_count = pdfs_collection.count_documents(query)
        
        # Calculate pagination
        skip = (page - 1) * per_page
        total_pages = (total_count + per_page - 1) // per_page
        
        # Fetch PDFs with pagination
        pdfs_cursor = pdfs_collection.find(query).sort('upload_timestamp', DESCENDING).skip(skip).limit(per_page)
        
        pdfs = []
        for pdf in pdfs_cursor:
            pdfs.append({
                'id': str(pdf['_id']),
                'filename': pdf['filename'],
                'original_filename': pdf['original_filename'],
                's3_key': pdf['s3_key'],
                's3_url': pdf_handler.get_s3_url_from_key(pdf['s3_key']),
                'storage_type': pdf['storage_type'],
                'mime_type': pdf['mime_type'],
                'file_size': pdf['file_size'],
                'upload_timestamp': pdf['upload_timestamp'].isoformat(),
                'is_redacted': pdf.get('is_redacted', False),
                'parent_id': pdf.get('parent_id')
            })
        
        return jsonify({
            'success': True,
            'pdfs': pdfs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'search_query': search_query
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdfs/<pdf_id>', methods=['GET'])
def get_pdf_details(pdf_id):
    """Get details of a specific PDF"""
    try:
        if not mongo_available:
            return jsonify({'error': 'Database not available'}), 503
        
        pdfs_collection = mongo_db['pdfs']
        pdf = pdfs_collection.find_one({'_id': ObjectId(pdf_id)})
        
        if not pdf:
            return jsonify({'error': 'PDF not found'}), 404
        
        return jsonify({
            'success': True,
            'pdf': {
                'id': str(pdf['_id']),
                'filename': pdf['filename'],
                'original_filename': pdf['original_filename'],
                's3_key': pdf['s3_key'],
                's3_url': pdf_handler.get_s3_url_from_key(pdf['s3_key']),
                'storage_type': pdf['storage_type'],
                'mime_type': pdf['mime_type'],
                'file_size': pdf['file_size'],
                'upload_timestamp': pdf['upload_timestamp'].isoformat(),
                'is_redacted': pdf.get('is_redacted', False),
                'parent_id': pdf.get('parent_id')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting PDF details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdfs/<pdf_id>', methods=['DELETE'])
def delete_pdf(pdf_id):
    """Soft delete a PDF (mark as inactive)"""
    try:
        if not mongo_available:
            return jsonify({'error': 'Database not available'}), 503
        
        pdfs_collection = mongo_db['pdfs']
        result = pdfs_collection.update_one(
            {'_id': ObjectId(pdf_id)},
            {'$set': {'status': 'deleted', 'deleted_at': datetime.utcnow()}}
        )
        
        if result.modified_count == 0:
            return jsonify({'error': 'PDF not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'PDF deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    health_info = {
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        's3_available': s3_available,
        's3_client_initialized': s3_client is not None,
        'mongodb_available': mongo_available,
        'mongodb_connected': mongo_client is not None,
        'local_storage': LOCAL_STORAGE_PATH,
        'aws_region': AWS_REGION,
        'bucket_name': S3_BUCKET,
        'database_name': MONGO_DB_NAME,
        'credentials_present': {
            'access_key': AWS_ACCESS_KEY is not None,
            'secret_key': AWS_SECRET_KEY is not None
        }
    }
    
    if s3_initialization_error:
        health_info['s3_error'] = s3_initialization_error
    
    return jsonify(health_info), 200

@app.route('/api/test-s3', methods=['GET'])
def test_s3_connection():
    """Test S3 connection endpoint"""
    if not s3_available:
        return jsonify({
            'success': False,
            'error': s3_initialization_error or 'S3 not available',
            'details': {
                'credentials_present': AWS_ACCESS_KEY is not None and AWS_SECRET_KEY is not None,
                'bucket_name': S3_BUCKET,
                'region': AWS_REGION
            }
        }), 400
    
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)
        
        return jsonify({
            'success': True,
            'message': 'S3 connection successful',
            'bucket': S3_BUCKET,
            'region': AWS_REGION,
            'object_count_sample': response.get('KeyCount', 0)
        }), 200
        
    except ClientError as e:
        return jsonify({
            'success': False,
            'error': f'S3 connection test failed: {e}',
            'error_code': e.response['Error']['Code']
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error testing S3: {e}'
        }), 500

@app.route('/api/test-mongodb', methods=['GET'])
def test_mongodb_connection():
    """Test MongoDB connection endpoint"""
    if not mongo_available:
        return jsonify({
            'success': False,
            'error': 'MongoDB not available'
        }), 400
    
    try:
        # Test connection with ping
        mongo_client.admin.command('ping')
        
        # Get collection stats
        pdfs_collection = mongo_db['pdfs']
        doc_count = pdfs_collection.count_documents({})
        
        return jsonify({
            'success': True,
            'message': 'MongoDB connection successful',
            'database': MONGO_DB_NAME,
            'total_pdfs': doc_count
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'MongoDB connection test failed: {e}'
        }), 500

@app.route('/api/test-redaction', methods=['POST'])
def test_redaction():
    """Test endpoint to verify redaction works properly"""
    try:
        if 'pdf' not in request.files or 'searchTerm' not in request.form:
            return jsonify({'success': False, 'error': 'Missing PDF file or search term'}), 400
        
        file = request.files['pdf']
        search_term = request.form['searchTerm'].strip()
        
        pdf_content = file.read()
        search_results = pdf_handler.search_word_in_pdf(pdf_content, search_term)
        
        redacted_pdf = pdf_handler.true_redaction_pdf(pdf_content, search_results, search_term)
        
        verification_results = pdf_handler.search_word_in_pdf(redacted_pdf, search_term)
        
        return jsonify({
            'success': True,
            'original_matches': len(search_results),
            'redacted_matches': len(verification_results),
            'redaction_successful': len(verification_results) == 0,
            'message': 'Redaction test completed' if len(verification_results) == 0 else 'Warning: Some text may still be recoverable'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search-suggestions', methods=['GET'])
def search_suggestions():
    """Get search suggestions based on existing filenames"""
    try:
        if not mongo_available:
            return jsonify({'error': 'Database not available'}), 503
        
        query = request.args.get('q', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({'suggestions': []}), 200
        
        pdfs_collection = mongo_db['pdfs']
        
        # Find matching filenames (limit to 10 suggestions)
        search_pattern = {'$regex': f'.*{re.escape(query)}.*', '$options': 'i'}
        
        suggestions = pdfs_collection.find(
            {
                'status': 'active',
                '$or': [
                    {'filename': search_pattern},
                    {'original_filename': search_pattern}
                ]
            },
            {'filename': 1, 'original_filename': 1}
        ).limit(10)
        
        suggestion_list = []
        seen = set()
        for doc in suggestions:
            for field in ['filename', 'original_filename']:
                if field in doc and doc[field] not in seen:
                    suggestion_list.append(doc[field])
                    seen.add(doc[field])
        
        return jsonify({
            'suggestions': suggestion_list[:10]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about PDFs"""
    try:
        if not mongo_available:
            return jsonify({'error': 'Database not available'}), 503
        
        pdfs_collection = mongo_db['pdfs']
        
        total_pdfs = pdfs_collection.count_documents({'status': 'active'})
        total_redacted = pdfs_collection.count_documents({'status': 'active', 'is_redacted': True})
        total_original = pdfs_collection.count_documents({'status': 'active', 'is_redacted': False})
        
        # Calculate total storage used
        pipeline = [
            {'$match': {'status': 'active'}},
            {'$group': {'_id': None, 'total_size': {'$sum': '$file_size'}}}
        ]
        size_result = list(pdfs_collection.aggregate(pipeline))
        total_size = size_result[0]['total_size'] if size_result else 0
        
        # Get recent uploads (last 7 days)
        seven_days_ago = datetime.utcnow() - __import__('datetime').timedelta(days=7)
        recent_uploads = pdfs_collection.count_documents({
            'status': 'active',
            'upload_timestamp': {'$gte': seven_days_ago}
        })
        
        return jsonify({
            'success': True,
            'stats': {
                'total_pdfs': total_pdfs,
                'total_redacted': total_redacted,
                'total_original': total_original,
                'total_storage_bytes': total_size,
                'total_storage_mb': round(total_size / (1024 * 1024), 2),
                'recent_uploads_7days': recent_uploads
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=== PDF Redaction Server Starting ===")
    print(f"Local storage path: {LOCAL_STORAGE_PATH}")
    print(f"S3 available: {s3_available}")
    if s3_available:
        print(f"S3 bucket: {S3_BUCKET}")
        print(f"S3 region: {AWS_REGION}")
    else:
        print(f"S3 initialization error: {s3_initialization_error}")
    print(f"MongoDB available: {mongo_available}")
    if mongo_available:
        print(f"MongoDB database: {MONGO_DB_NAME}")
    print("=== Server Ready ===")
    app.run(debug=True, host='0.0.0.0', port=5000)