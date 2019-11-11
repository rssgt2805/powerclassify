import os
import pandas as pd
from flask import Flask,render_template,session,url_for,redirect,logging,request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,scoped_session
from flask_migrate import Migrate
from passlib.hash import sha256_crypt
from final import *
from flask import flash

app = Flask(__name__)
engine = create_engine("mysql+pymysql://root:Chinmayc14!@#@localhost/registers")
db = scoped_session(sessionmaker(bind=engine))
un = []

#home
@app.route('/')
def index():
    return render_template('index.html')
#login

#register
@app.route('/register',methods=["GET","POST"])
def register():
    if request.method=="POST":
        name = request.form.get("name")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))
        if password == confirm:
            db.execute("INSERT INTO user(name,username,password) VALUES(:name,:username,:password)", {"name":name,"username":username,"password":secure_password})
            db.commit()

            return redirect(url_for('login'))
        else:
            flash(" Your Password do not match!")
            return render_template('register.html')
    return render_template('register.html')
@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=="POST":
        username = request.form.get("username")
        password = request.form.get("password")
        un.append(username)
        print(type(username))

        usernamedata = db.execute("SELECT username FROM user WHERE username=:username",{"username":username}).fetchone()
        passwordata = db.execute("SELECT password FROM user WHERE username=:username",{"username":username}).fetchone()
        print(passwordata)
        print(un)

        if usernamedata == None:

            return render_template('login.html')
        else:
            for passwor_data in passwordata:
                if sha256_crypt.verify(password , passwor_data):

                    return redirect(url_for('afterlogin'))
                else:

                    return render_template('login.html')
    return render_template('login.html')


#features
@app.route('/features')
def features():
    return render_template('features.html')
@app.route('/afterlogin')
def afterlogin():
    return render_template('afterlogin.html')
@app.route('/afterloginhome')
def afterloginhome():
    return render_template('afterloginhome.html')
@app.route('/ALyes')
def ALyes():
    return render_template('ALyes.html')
@app.route('/ALno')
def ALno():
    return render_template('ALno.html')
@app.route('/START')
def START():
    return render_template('START.html')
@app.route('/PM')
def PM():

    x = pd.read_csv('data_reg_final_4.csv')
    l = x.date
    c =  x.power
    mydict = {'min':l,'power': c}
    data1 = pd.DataFrame(mydict,columns =['min','power'])
    return render_template('PM.html',data = data1)
@app.route('/PC')
def PC():
    if request.method=="POST":
        name = request.form.get("name")

        days = request.form.get("days")
        db.execute("use registers INSERT INTO user(days) VALUES(:days) WHERE name=:name", {"days":days, "name":name})
        db.commit()
    return render_template('PC.html')

@app.route('/PR')
def PR():
    #daysd = db.execute("use registers SELECT days FROM user").fetchone()
    z = predict_days(7,'data_reg_final_4.csv',1440)
    return render_template('PR.html',data =z)


print(un)
if __name__ == '__main__':
    app.run(debug=True)
