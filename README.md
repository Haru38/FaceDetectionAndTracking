## Face detection and tracking
Face detection and tracking by openCV
## Face detection
- Face detection using opencv's cascadefile.
## Face tracking
- Track the detected faces using meanshift.
- This eliminates the need to detect faces every frame and speeds up tracking.
## firebase
- Add the coordinates of detected (tracked) faces to database via firebase.
- Do not slow down the face tracking process by doing it in a separate thread.
## Receive coordinates
- Receive face coordinates from firebase by javaScript.
## End goal
- Apply the face coordinates to the two-dimensional character in the screen, and make it work with the movement of your own face.

