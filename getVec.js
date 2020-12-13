var admin = require('firebase-admin');
var serviceAccount = require('./facetracking-67293-firebase-adminsdk-yvqw7-bee8b09b2d.json');

admin.initializeApp( {
    credential: admin.credential.cert(serviceAccount),
    databaseURL: "https://facetracking-67293.firebaseio.com/" //データベースのURL
} );

var db = admin.database();
db.ref('saikochan-info').on('child_added', function (obj) {
    console.log(obj.val());
});
db.ref('coordinate').on('child_added', function (obj) {
    console.log(obj.val());
});