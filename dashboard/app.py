from flask import Flask, render_template, send_from_directory
import os
import data_loader

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


@app.route('/')
def overview():
    return render_template('overview.html', active='overview',
                           **data_loader.get_overview_data())


@app.route('/attackers')
def attackers():
    return render_template('attackers.html', active='attackers',
                           **data_loader.get_attackers_data())


@app.route('/features')
def features():
    return render_template('features.html', active='features',
                           **data_loader.get_features_data())


@app.route('/matrices')
def matrices():
    return render_template('matrices.html', active='matrices',
                           **data_loader.get_matrices_data())


@app.route('/methodology')
def methodology():
    return render_template('methodology.html', active='methodology',
                           **data_loader.get_methodology_data())


@app.route('/story/phishing')
def story_phishing():
    return render_template('story_phishing.html', active='stories',
                           story_id='phishing',
                           **data_loader.get_story_phishing())


@app.route('/story/credential_stuffing')
def story_cred_stuffing():
    return render_template('story_cred_stuffing.html', active='stories',
                           story_id='credential_stuffing',
                           **data_loader.get_story_cred_stuffing())


@app.route('/story/planned_exfil')
def story_planned_exfil():
    return render_template('story_planned_exfil.html', active='stories',
                           story_id='planned_exfil',
                           **data_loader.get_story_planned_exfil())


@app.route('/data-plots/<path:filename>')
def data_plots(filename):
    return send_from_directory(DATA_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
