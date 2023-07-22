import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:dio/dio.dart';
import 'package:animated_text_kit/animated_text_kit.dart';

late List<CameraDescription> _cameras = List.empty();
final url = Uri.parse(
    'https://us-central1-cp301-plant-disease-prediction.cloudfunctions.net/predict');

void main() {
  runApp(MaterialApp(
    home: SplashScreen(),
  ));
}

class SplashScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Create a delayed future that completes after 2 seconds
    Future.delayed(Duration(seconds: 3), () {
      // Navigate to the main content screen
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => Home()),
      );
    });

    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ClipOval(
              child: Image.asset(
                'assets/images/logo.png',
                width: 100,
                height: 100,
              ),
            ),
            SizedBox(height: 20),
            TyperAnimatedTextKit(
              speed: Duration(milliseconds: 300),
              totalRepeatCount: 1,
              text: ['PlantDoc'],
              textStyle: TextStyle(
                fontSize: 25.0,
                fontWeight: FontWeight.bold,
                fontFamily: 'Roboto',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class MainContentScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    Color myCustomColor = Color.fromARGB(255, 155, 173, 61);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: myCustomColor,
        title: Text('Main Content'),
      ),
      body: Center(
        child: Text('This is the main content screen'),
      ),
    );
  }
}

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  XFile? image;

  final ImagePicker picker = ImagePicker();

  var _gettingPrediction = 0;

  late String class1, class2;
  late num conf;
  late bool reclick;

  List<dynamic> getFields(Map<String, dynamic> json) {
    bool reclick = json['click_again'];
    String class1 = "NULL", class2 = "NULL";
    num conf = 0;

    if (reclick == true) {
      class1 = (json['class'][0]);
      class2 = (json['class'][1]);
    } else {
      conf = (json['confidence']);
      class1 = (json['class']);
    }

    return [class1, class2, conf, reclick];
  }

  //we can upload image from camera or from gallery based on parameter
  Future getImage(ImageSource media) async {
    var img = await picker.pickImage(source: media);

    setState(() {
      image = img;
    });
  }

  Future<Map<String, dynamic>> GetPrediction(File image) async {
    setState(() {
      _gettingPrediction = 1;
    });

    var request = new http.MultipartRequest("POST", url);

    // var stream = new http.ByteStream(image.openRead());
    // stream.cast();

    var multipart = await http.MultipartFile.fromPath("file", image.path);

    // FormData data = FormData.fromMap({'': '', 'file': multipart});

    request.files.add(multipart);

    // request.fields['file'] = "file";

    var response = await request.send();
    // var response = await dio.post(url.toString(), data: data);

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      print(respStr);
      return jsonDecode(respStr);
    } else {
      throw Exception('Failed to connect to server\n Connection Error :(');
    }
  }

  //show popup dialog
  void myAlert() {
    showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            title: Text('Please choose media to select'),
            content: Container(
              height: MediaQuery.of(context).size.height / 6,
              child: Column(
                children: [
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      getImage(ImageSource.gallery);
                    },
                    child: Row(
                      children: [
                        Icon(Icons.image),
                        Text('From Gallery'),
                      ],
                    ),
                    style: ElevatedButton.styleFrom(
                        primary: Color.fromARGB(255, 98, 118, 30)),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      getImage(ImageSource.camera);
                    },
                    child: Row(
                      children: [
                        Icon(Icons.camera),
                        Text('From Camera'),
                      ],
                    ),
                    style: ElevatedButton.styleFrom(
                        primary: Color.fromARGB(255, 98, 118, 30)),
                  ),
                ],
              ),
            ),
          );
        });
  }

  @override
  Widget build(BuildContext context) {
    Color myCustomColor = Color.fromARGB(255, 155, 173, 61);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: myCustomColor,
        title: Text('Crop Disease Prediction'),
      ),
      body: Center(
        child: _gettingPrediction > 0
            ? (_gettingPrediction == 1
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                          padding: EdgeInsets.only(bottom: 10),
                          child: Text('Fetching Results!')),
                      CircularProgressIndicator(),
                    ],
                  )
                : (reclick == true
                    ? Center(
                        child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                  padding: EdgeInsets.only(bottom: 10),
                                  child: Text('Predicted Classes: ')),
                              Text(
                                class1 + ', ' + class2 + '\n',
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                              Container(
                                padding: EdgeInsets.only(top: 20, bottom: 10),
                                child: Text(
                                    'Consider Reclicking the photo or reuploading!',
                                    style:
                                        TextStyle(fontStyle: FontStyle.italic)),
                              ),
                              Padding(
                                padding:
                                    const EdgeInsets.symmetric(horizontal: 20),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(8),
                                  child: Image.file(
                                    File(image!.path),
                                    fit: BoxFit.cover,
                                    width: MediaQuery.of(context).size.width,
                                    height: 300,
                                  ),
                                ),
                              ),
                              ElevatedButton(
                                  onPressed: () {
                                    setState(() {
                                      _gettingPrediction = 0;
                                      image = null;
                                    });
                                  },
                                  child: Text('OK'))
                            ]),
                      )
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('Predicted Class: '),
                            Container(
                              padding: EdgeInsets.only(top: 10, bottom: 10),
                              child: Text(
                                class1,
                                textAlign: TextAlign.center,
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                            ),
                            Text('Confidence: '),
                            Container(
                              padding: EdgeInsets.only(top: 10, bottom: 10),
                              child: Text(
                                conf.toString(),
                                textAlign: TextAlign.center,
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                            ),
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 20),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8),
                                child: Image.file(
                                  File(image!.path),
                                  fit: BoxFit.cover,
                                  width: MediaQuery.of(context).size.width,
                                  height: 300,
                                ),
                              ),
                            ),
                            ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    _gettingPrediction = 0;
                                    image = null;
                                  });
                                },
                                child: Text('OK'))
                          ],
                        ),
                      )))
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      image == null
                          ? myAlert()
                          : GetPrediction(File(image!.path)).then((value) {
                              if (value['class'] == 'none') {
                                showDialog(
                                    context: context,
                                    builder: (BuildContext context) {
                                      return AlertDialog(
                                        title: Text("Connection Error! Oops"),
                                        content: Container(
                                          padding: EdgeInsets.symmetric(
                                              vertical: 10),
                                          child: ElevatedButton(
                                            child: Text('OK'),
                                            onPressed: () => setState(() {
                                              image = null;
                                              _gettingPrediction = 0;
                                            }),
                                          ),
                                        ),
                                      );
                                    });
                              } else {
                                setState(() {
                                  var val = getFields(value);
                                  class1 = val[0];
                                  class2 = val[1];
                                  conf = val[2];
                                  reclick = val[3];
                                  _gettingPrediction = 2;
                                });
                              }
                            });
                    },
                    child: Text('Upload Photo'),
                    style: ElevatedButton.styleFrom(
                        primary: Color.fromARGB(255, 155, 173, 61)),
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  //if image not null show the image
                  //if image null show text
                  image != null
                      ? Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 20),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: Image.file(
                              File(image!.path),
                              fit: BoxFit.cover,
                              width: MediaQuery.of(context).size.width,
                              height: 300,
                            ),
                          ),
                        )
                      : Text(
                          "No Image",
                          style: TextStyle(fontSize: 20),
                        )
                ],
              ),
      ),
    );
  }
}
