import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

"""
firebaseの初期設定
info_ref:スクリーンサイズ
coordinate_ref:顔の座標
smile_ref:笑っているか否かのフラグ
"""


#firebaseの前処理
cred = credentials.Certificate('######.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "##########",
    'databaseAuthVariableOverride': {
        'uid': "########"
    }
})
info_ref = db.reference("###########")
coordinate_ref = db.reference('#######')
#smile_ref = db.refernce('/smile')