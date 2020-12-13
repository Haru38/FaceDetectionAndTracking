var admin = require('firebase-admin');
var serviceAccount = require('#######.json');

admin.initializeApp( {
    credential: admin.credential.cert(serviceAccount),
    databaseURL: "#######" //データベースのURL
} );

var db = admin.database();
db.ref('info').on('child_added', function (obj) {
    console.log(obj.val());
});
db.ref('coordinate').on('child_added', function (obj) {
    console.log(obj.val());
});
