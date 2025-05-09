from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from yourapplication import db
from yourapplication.models import MathProgress
from datetime import datetime

visione_bp = Blueprint('visione', __name__)

# Keep only visione-specific routes here
@visione_bp.route('/')
def visione():
    return render_template('class_img_2.html')
