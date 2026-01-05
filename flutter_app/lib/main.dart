import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // Put your deployed Streamlit URL here
  // For Android emulator local dev, often: http://10.0.2.2:8501
  final String streamlitUrl = "https://YOUR-STREAMLIT-APP-URL";

  @override
  Widget build(BuildContext context) {
    final controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..loadRequest(Uri.parse(streamlitUrl));

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text("YOLO Detector")),
        body: WebViewWidget(controller: controller),
      ),
    );
  }
}
