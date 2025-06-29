from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os
import json
import secrets
from functools import wraps
import sqlalchemy as sa

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# UPDATED: Load the improved model and encoders
def load_model_resources():
    try:
        with open('workplace_stress_encoders_improved.pkl', 'rb') as f:
            encoders_info = pickle.load(f)

        # Using the ensemble model for better accuracy
        with open('workplace_stress_ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)

        return encoders_info, model
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        # If improved model files not found, try loading original model files
        try:
            with open('workplace_stress_encoders.pkl', 'rb') as f:
                encoders_info = pickle.load(f)

            with open('workplace_stress_xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)

            return encoders_info, model
        except FileNotFoundError:
            raise Exception("No model files found. Please ensure model files are in the correct location.")


# Database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    stress_records = db.relationship('StressRecord', backref='user', lazy=True)


class StressRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    sleep_hours = db.Column(db.Float, nullable=False)
    body_temperature = db.Column(db.String(50), nullable=False)
    noise_levels = db.Column(db.String(50), nullable=False)
    working_hours = db.Column(db.String(50), nullable=False)
    working_area_temperature = db.Column(db.String(50), nullable=False)
    workload = db.Column(db.String(50), nullable=False)
    type_of_work = db.Column(db.String(50), nullable=False)
    working_shift = db.Column(db.String(50), nullable=False)
    stress_level = db.Column(db.String(50), nullable=False)
    stress_percentage = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), nullable=True, default="improved")  # Made nullable for compatibility


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)

    return decorated_function


# Routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Simple validation
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))

        # Check if username or email already exists
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists', 'danger')
            return redirect(url_for('signup'))

        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)

        # Set admin status (for testing purposes - first user as admin)
        if User.query.count() == 0:
            new_user.is_admin = True

        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)

            # Redirect based on user role
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/stress-predictor', methods=['GET', 'POST'])
def stress_predictor():
    try:
        # Load model resources
        encoders_info, model = load_model_resources()

        # Get option lists for dropdowns - UPDATED for new categories
        body_temp_options = list(encoders_info['cat_mappings']['body_temperature'].keys())
        noise_options = list(encoders_info['cat_mappings']['noise_levels'].keys())
        work_hours_options = list(encoders_info['cat_mappings']['working_hours'].keys())
        temp_options = list(encoders_info['cat_mappings']['working_area_temperature'].keys())
        workload_options = list(encoders_info['cat_mappings']['workload'].keys())
        work_type_options = list(encoders_info['cat_mappings']['type_of_work'].keys())
        shift_options = list(encoders_info['cat_mappings']['working_shift'].keys())

        if request.method == 'POST':
            # Get form data
            sleep_hours = float(request.form.get('sleep_hours'))
            body_temperature = request.form.get('body_temperature')
            noise_levels = request.form.get('noise_levels')
            working_hours = request.form.get('working_hours')
            working_area_temperature = request.form.get('working_area_temperature')
            workload = request.form.get('workload')
            type_of_work = request.form.get('type_of_work')
            working_shift = request.form.get('working_shift')

            # Input data
            input_data = {
                'sleep_hours': sleep_hours,
                'body_temperature': body_temperature,
                'noise_levels': noise_levels,
                'working_hours': working_hours,
                'working_area_temperature': working_area_temperature,
                'workload': workload,
                'type_of_work': type_of_work,
                'working_shift': working_shift
            }

            # Make prediction with UPDATED function
            pred_label, pred_encoded, pred_proba = predict_stress_level(input_data, encoders_info, model)

            # Calculate percentage
            max_prob = max(pred_proba) * 100

            # Save record if user is logged in
            if current_user.is_authenticated:
                # Check if model_version column exists
                try:
                    stress_record = StressRecord(
                        user_id=current_user.id,
                        sleep_hours=sleep_hours,
                        body_temperature=body_temperature,
                        noise_levels=noise_levels,
                        working_hours=working_hours,
                        working_area_temperature=working_area_temperature,
                        workload=workload,
                        type_of_work=type_of_work,
                        working_shift=working_shift,
                        stress_level=pred_label,
                        stress_percentage=max_prob,
                        model_version="improved"  # Track which model version was used
                    )
                except Exception:
                    # If model_version column doesn't exist
                    stress_record = StressRecord(
                        user_id=current_user.id,
                        sleep_hours=sleep_hours,
                        body_temperature=body_temperature,
                        noise_levels=noise_levels,
                        working_hours=working_hours,
                        working_area_temperature=working_area_temperature,
                        workload=workload,
                        type_of_work=type_of_work,
                        working_shift=working_shift,
                        stress_level=pred_label,
                        stress_percentage=max_prob
                    )

                db.session.add(stress_record)
                db.session.commit()

            # Get appropriate message based on stress level
            messages = {
                "Low": "Your stress level is low. Keep up the healthy balance! 😊",
                "Medium": "Your stress level is moderate. Consider taking short breaks during work. 🙂",
                "High": "Your stress level is high. Try some stress management techniques. 😐",
                "Extremely High": "Your stress level is extremely high! Please consider consulting a professional. 😨"
            }

            message = messages.get(pred_label, "Result unavailable")

            # Return prediction result
            return render_template('stress_predictor.html',
                                   body_temp_options=body_temp_options,
                                   noise_options=noise_options,
                                   work_hours_options=work_hours_options,
                                   temp_options=temp_options,
                                   workload_options=workload_options,
                                   work_type_options=work_type_options,
                                   shift_options=shift_options,
                                   prediction=pred_label,
                                   percentage=f"{max_prob:.2f}%",
                                   message=message,
                                   input_data=input_data,
                                   show_result=True)

        return render_template('stress_predictor.html',
                               body_temp_options=body_temp_options,
                               noise_options=noise_options,
                               work_hours_options=work_hours_options,
                               temp_options=temp_options,
                               workload_options=workload_options,
                               work_type_options=work_type_options,
                               shift_options=shift_options,
                               show_result=False)

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return render_template('error.html', error=str(e))


@app.route('/user-dashboard')
@login_required
def user_dashboard():
    # Get user's stress records ordered by date (newest first)
    stress_records = StressRecord.query.filter_by(user_id=current_user.id).order_by(StressRecord.date.desc()).all()
    return render_template('user_dashboard.html', stress_records=stress_records)


@app.route('/admin-dashboard')
@login_required
@admin_required
def admin_dashboard():
    # Get all users except current admin
    users = User.query.filter(User.id != current_user.id).all()
    # Get all stress records
    stress_records = StressRecord.query.order_by(StressRecord.date.desc()).all()

    # Count metrics
    total_users = User.query.count() - 1  # Excluding admin
    total_records = StressRecord.query.count()

    # Stress level distribution
    stress_distribution = db.session.query(
        StressRecord.stress_level,
        db.func.count(StressRecord.id)
    ).group_by(StressRecord.stress_level).all()

    stress_levels = [level for level, count in stress_distribution]
    stress_counts = [count for level, count in stress_distribution]

    return render_template('admin_dashboard.html',
                           users=users,
                           stress_records=stress_records,
                           total_users=total_users,
                           total_records=total_records,
                           stress_levels=json.dumps(stress_levels),
                           stress_counts=json.dumps(stress_counts))


@app.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    record = StressRecord.query.get_or_404(record_id)

    # Check if the record belongs to current user or user is admin
    if record.user_id != current_user.id and not current_user.is_admin:
        flash("You don't have permission to view this record", "danger")
        return redirect(url_for('user_dashboard'))

    # Handle the case where model_version might not exist
    try:
        model_version = record.model_version
    except:
        model_version = "original"

    return jsonify({
        'id': record.id,
        'date': record.date.strftime('%Y-%m-%d %H:%M'),
        'sleep_hours': record.sleep_hours,
        'body_temperature': record.body_temperature,
        'noise_levels': record.noise_levels,
        'working_hours': record.working_hours,
        'working_area_temperature': record.working_area_temperature,
        'workload': record.workload,
        'type_of_work': record.type_of_work,
        'working_shift': record.working_shift,
        'stress_level': record.stress_level,
        'stress_percentage': f"{record.stress_percentage:.2f}%",
        'model_version': model_version
    })


@app.route('/suggestions')
@login_required
def suggestions():
    # Get user's latest stress record
    latest_record = StressRecord.query.filter_by(user_id=current_user.id).order_by(StressRecord.date.desc()).first()

    if not latest_record:
        flash("No stress records found. Please complete a stress prediction first.", "info")
        return redirect(url_for('stress_predictor'))

    # Generate suggestions based on stress level
    suggestions = get_suggestions(latest_record)

    return render_template('suggestions.html',
                           record=latest_record,
                           suggestions=suggestions)


@app.route('/analysis')
@login_required
def analysis():
    # Get user's stress records
    stress_records = StressRecord.query.filter_by(user_id=current_user.id).order_by(StressRecord.date).all()

    if not stress_records:
        flash("No stress records found. Please complete a stress prediction first.", "info")
        return redirect(url_for('stress_predictor'))

    # Prepare data for charts
    dates = [record.date.strftime('%Y-%m-%d') for record in stress_records]
    stress_levels_numeric = []

    # Convert stress levels to numeric for graphing
    for record in stress_records:
        if record.stress_level == "Low":
            stress_levels_numeric.append(0)
        elif record.stress_level == "Medium":
            stress_levels_numeric.append(1)
        elif record.stress_level == "High":
            stress_levels_numeric.append(2)
        else:  # Extremely High
            stress_levels_numeric.append(3)

    # Count occurrences of each stress level
    stress_level_counts = {}
    for record in stress_records:
        if record.stress_level in stress_level_counts:
            stress_level_counts[record.stress_level] += 1
        else:
            stress_level_counts[record.stress_level] = 1

    # Factor analysis
    workload_impact = {}
    sleep_impact = {}

    for record in stress_records:
        # Workload impact
        if record.workload not in workload_impact:
            workload_impact[record.workload] = {"count": 0, "stress_sum": 0}

        workload_impact[record.workload]["count"] += 1

        # Convert stress level to numeric for averaging
        stress_value = 0
        if record.stress_level == "Low":
            stress_value = 0
        elif record.stress_level == "Medium":
            stress_value = 1
        elif record.stress_level == "High":
            stress_value = 2
        else:  # Extremely High
            stress_value = 3

        workload_impact[record.workload]["stress_sum"] += stress_value

        # Sleep hours impact - group in ranges
        sleep_range = ""
        if record.sleep_hours < 5:
            sleep_range = "< 5 hours"
        elif record.sleep_hours < 6:
            sleep_range = "5-6 hours"
        elif record.sleep_hours < 7:
            sleep_range = "6-7 hours"
        elif record.sleep_hours < 8:
            sleep_range = "7-8 hours"
        else:
            sleep_range = "> 8 hours"

        if sleep_range not in sleep_impact:
            sleep_impact[sleep_range] = {"count": 0, "stress_sum": 0}

        sleep_impact[sleep_range]["count"] += 1
        sleep_impact[sleep_range]["stress_sum"] += stress_value

    # Calculate average stress for each factor
    for category in workload_impact:
        if workload_impact[category]["count"] > 0:
            workload_impact[category]["avg_stress"] = workload_impact[category]["stress_sum"] / \
                                                      workload_impact[category]["count"]
        else:
            workload_impact[category]["avg_stress"] = 0

    for category in sleep_impact:
        if sleep_impact[category]["count"] > 0:
            sleep_impact[category]["avg_stress"] = sleep_impact[category]["stress_sum"] / sleep_impact[category][
                "count"]
        else:
            sleep_impact[category]["avg_stress"] = 0

    # Sort sleep impacts by hours
    sleep_order = ["< 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "> 8 hours"]
    sorted_sleep_impact = {k: sleep_impact[k] for k in sleep_order if k in sleep_impact}

    return render_template('analysis.html',
                           dates=json.dumps(dates),
                           stress_levels=json.dumps(stress_levels_numeric),
                           stress_level_counts=stress_level_counts,
                           workload_impact=workload_impact,
                           sleep_impact=sorted_sleep_impact)


# FIXED: Improved predict_stress_level function that ensures feature compatibility
def predict_stress_level(input_data, encoders_info, model):
    """Make stress level prediction with improved model features"""
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Print for debugging - remove in production
    print("Expected feature columns:", encoders_info['feature_columns'])

    # Create engineered features from the input data
    input_df['sleep_squared'] = input_df['sleep_hours'] ** 2
    input_df['sleep_low'] = (input_df['sleep_hours'] < 6).astype(int)
    input_df['sleep_optimal'] = ((input_df['sleep_hours'] >= 7) &
                                 (input_df['sleep_hours'] <= 8)).astype(int)

    input_df['night_noisy'] = ((input_df['working_shift'] == 'Night shift') &
                               (input_df['noise_levels'] == 'Noisy')).astype(int)

    input_df['bad_conditions'] = ((input_df['working_area_temperature'] == 'Uncomfortable') &
                                  (input_df['type_of_work'] == 'Demanding')).astype(int)

    input_df['overworked'] = ((input_df['workload'] == 'Heavy') &
                              (input_df['working_hours'].isin(['Long Day', 'Extreme Overtime']))).astype(int)

    input_df['sleep_workload_interaction'] = input_df.apply(
        lambda row: 1 if (row['sleep_hours'] < 6 and row['workload'] == 'Heavy') else 0,
        axis=1
    )

    # Count stress factors
    stress_conditions = [
        input_df['sleep_low'] == 1,
        input_df['working_shift'] == 'Night shift',
        input_df['workload'] == 'Heavy',
        input_df['noise_levels'] == 'Noisy',
        input_df['working_area_temperature'] == 'Uncomfortable',
        input_df['type_of_work'] == 'Demanding'
    ]
    input_df['stress_factors_count'] = sum(stress_conditions)

    # Encode categorical features first
    for col in encoders_info['cat_mappings'].keys():
        le = encoders_info['label_encoders'][col]
        input_df[col + '_encoded'] = le.transform(input_df[col])

    # For each feature column expected by the model, ensure it exists
    result_df = pd.DataFrame(index=input_df.index)

    for feature in encoders_info['feature_columns']:
        if feature in input_df.columns:
            # If it's a numeric feature that needs scaling
            if feature in ['sleep_hours', 'sleep_squared', 'stress_factors_count']:
                # Scale each feature individually
                feature_values = input_df[[feature]].values
                scaled_values = encoders_info['numeric_scaler'].transform(feature_values)
                result_df[feature] = scaled_values.flatten()
            else:
                # Copy binary and categorical features as is
                result_df[feature] = input_df[feature]
        else:
            print(f"Missing feature: {feature}")
            raise ValueError(f"Required feature '{feature}' is missing")

    # Make prediction
    pred_encoded = model.predict(result_df)[0]

    # Handle probability prediction for both single and ensemble models
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(result_df)[0]
    else:
        # Some ensemble methods might handle probabilities differently
        pred_proba = np.zeros(len(encoders_info['label_encoders']['calculated_stress_level'].classes_))
        pred_proba[pred_encoded] = 1.0

    # Decode prediction
    pred_label = encoders_info['label_encoders']['calculated_stress_level'].inverse_transform([pred_encoded])[0]

    return pred_label, pred_encoded, pred_proba


# UPDATED: Get suggestions based on new categories
def get_suggestions(stress_record):
    """Generate suggestions based on stress level and factors - updated for new model categories"""
    suggestions = []

    # Base suggestions based on stress level
    if stress_record.stress_level == "Low":
        suggestions.append({
            "title": "Maintain Your Routine",
            "content": "Your current routine is working well. Continue with your current sleep schedule and work habits."
        })
    elif stress_record.stress_level == "Medium":
        suggestions.append({
            "title": "Regular Breaks",
            "content": "Take short 5-minute breaks every hour to refresh your mind and reduce accumulated stress."
        })
    elif stress_record.stress_level == "High":
        suggestions.append({
            "title": "Stress Management Techniques",
            "content": "Practice deep breathing exercises and consider short meditation sessions during your day."
        })
    else:  # Extremely High
        suggestions.append({
            "title": "Professional Support",
            "content": "Consider speaking to a mental health professional. Your stress levels are concerning and may benefit from professional guidance."
        })

    # Sleep-based suggestions
    if stress_record.sleep_hours < 6:
        suggestions.append({
            "title": "Improve Sleep Quality",
            "content": f"You're only getting {stress_record.sleep_hours} hours of sleep. Try to increase to at least 7 hours by establishing a consistent sleep schedule."
        })

    # Workload suggestions - updated categories
    if stress_record.workload == "Heavy":
        suggestions.append({
            "title": "Workload Management",
            "content": "Your workload is high. Consider delegating tasks or discussing priorities with your supervisor."
        })

    # Temperature comfort suggestions - updated categories
    if stress_record.working_area_temperature == "Uncomfortable":
        suggestions.append({
            "title": "Adjust Your Environment",
            "content": "Your workspace is uncomfortable. If possible, adjust temperature settings or use personal solutions like desk fans or warm clothing."
        })

    # Noise level suggestions (unchanged)
    if stress_record.noise_levels == "Noisy":
        suggestions.append({
            "title": "Manage Noise",
            "content": "Consider noise-cancelling headphones or requesting a quieter workspace if available."
        })

    # Working hours suggestions - updated categories
    if stress_record.working_hours in ["Long Day", "Extreme Overtime"]:
        suggestions.append({
            "title": "Work-Life Balance",
            "content": "You're working long hours regularly. Try to establish clearer boundaries between work and personal time."
        })

    # Night shift suggestions (unchanged)
    if stress_record.working_shift == "Night shift":
        suggestions.append({
            "title": "Night Shift Adaptation",
            "content": "Night shifts can disrupt your circadian rhythm. Ensure your sleeping area is dark during day sleep, and consider blackout curtains."
        })

    # Type of work suggestions - updated categories
    if stress_record.type_of_work == "Demanding":
        suggestions.append({
            "title": "Manage Demanding Work",
            "content": "Your work is demanding. Consider techniques like time-blocking and regular short breaks to maintain focus and reduce mental fatigue."
        })

    # Body temperature suggestions - updated categories
    if stress_record.body_temperature == "Not Normal":
        suggestions.append({
            "title": "Monitor Your Health",
            "content": "Your body temperature is not normal. Consider checking for signs of illness or monitor workplace conditions that might be affecting your temperature."
        })

    return suggestions


# Database setup with column check
def check_and_update_db_schema():
    # Check if the model_version column exists in the stress_record table
    with app.app_context():
        inspector = sa.inspect(db.engine)
        columns = [column['name'] for column in inspector.get_columns('stress_record')]

        # If table doesn't exist yet, just create it normally
        if 'stress_record' not in inspector.get_table_names():
            db.create_all()
            return

        # If column doesn't exist, add it
        if 'model_version' not in columns:
            try:
                db.engine.execute('ALTER TABLE stress_record ADD COLUMN model_version VARCHAR(50)')
                print("Added model_version column to stress_record table")
            except Exception as e:
                print(f"Error adding model_version column: {e}")
                # For SQLite, which has limited ALTER TABLE support, this might fail
                # In production, you would use a migration framework like Flask-Migrate


# Initialize database
with app.app_context():
    db.create_all()

    # Check and update schema if needed
    check_and_update_db_schema()

    # Create admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_password = generate_password_hash('admin123', method='pbkdf2:sha256')
        admin = User(username='admin', email='admin@stressdetection.com', password=admin_password, is_admin=True)
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True)