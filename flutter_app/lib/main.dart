import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // Change this to your Streamlit URL:
  // - Android emulator (local dev): http://10.0.2.2:8501
  // - Real device on Wi-Fi: http://192.168.x.x:8501
  // - Production: https://your-app-name.streamlit.app
  final String streamlitUrl = "http://10.0.2.2:8501";

  @override
  Widget build(BuildContext context) {
    final controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFFFFFFFF))
      ..setNavigationDelegate(
        NavigationDelegate(
          onWebResourceError: (error) {
            debugPrint("WebView error: ${error.description}");
          },
        ),
      )
      ..loadRequest(Uri.parse(streamlitUrl));

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text("YOLO Detector")),
        body: WebViewWidget(controller: controller),
      ),
    );
  }
}
