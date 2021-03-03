from flask import Blueprint, render_template, request, redirect, url_for, session

views = Blueprint('views',__name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['imgURL'] = request.form.get('imgURL')
        session['calcState'] = request.form.get('calcState')
        print(session.get('imgURL'))
        return redirect(url_for('views.edit'))
    else:
        return render_template('home.html')

@views.route('/edit')
def edit():
    print(session.get('calcState'))
    return "<h1>test</h1>"