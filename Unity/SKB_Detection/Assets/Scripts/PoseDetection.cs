using UnityEngine;

using System.Collections.Generic;

public class PoseDetection : MonoBehaviour
{
    [Header("Модель и камера")]
    public Unity.InferenceEngine.ModelAsset modelAsset;
    public WebCamTexture webCamTexture;

    [Header("Настройки")]
    public float confidenceThreshold = 0.5f;

    private Unity.InferenceEngine.Model runtimeModel;
    private Unity.InferenceEngine.Worker worker;
    private RenderTexture inputTexture;

    // Цвета для скелета
    private Color skeletonColor = Color.green;
    private Texture2D lineTexture;

    void Start()
    {
        // Запускаем камеру
        webCamTexture = new WebCamTexture(640, 640);
        webCamTexture.Play();

        // Загружаем модель
        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
        worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.GPUCompute);

        // Текстура для линий скелета
        lineTexture = new Texture2D(1, 1);
        lineTexture.SetPixel(0, 0, Color.green);
        lineTexture.Apply();

        inputTexture = new RenderTexture(640, 640, 0);

        Debug.Log("Модель загружена, камера запущена");
    }

    void Update()
    {
        if (webCamTexture == null || !webCamTexture.isPlaying) return;

        // Копируем кадр с камеры в RenderTexture 640x640
        Graphics.Blit(webCamTexture, inputTexture);

        // Подаём на вход модели
        using var inputTensor = Unity.InferenceEngine.TextureConverter.ToTensor(inputTexture, 640, 640, 3);
        worker.Schedule(inputTensor);

        // Получаем результат
        using var output = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        if (output == null) return;

        var data = output.DownloadToArray();
        // data имеет форму [1, 56, 8400]
        // 56 = 5 (x,y,w,h,conf) + 51 (17 keypoints * 3)
        ProcessDetections(data);
    }

    // Координаты keypoints для отрисовки
    private List<Vector2> keypoints = new List<Vector2>();

    void ProcessDetections(float[] data)
    {
        int numDetections = 8400;
        keypoints.Clear();

        for (int i = 0; i < numDetections; i++)
        {
            float conf = data[4 * 8400 + i];
            if (conf < confidenceThreshold) continue;

            // Извлекаем 17 keypoints
            for (int k = 0; k < 17; k++)
            {
                int baseIdx = (5 + k * 3);
                float kx = data[baseIdx * 8400 + i];
                float ky = data[(baseIdx + 1) * 8400 + i];
                float kconf = data[(baseIdx + 2) * 8400 + i];

                if (kconf > 0.3f)
                {
                    // Переводим в экранные координаты
                    float screenX = (kx / 640f) * Screen.width;
                    float screenY = (1f - ky / 640f) * Screen.height;
                    keypoints.Add(new Vector2(screenX, screenY));
                }
                else
                {
                    keypoints.Add(Vector2.zero);
                }
            }

            Debug.Log($"Человек обнаружен! Уверенность: {conf:F2}");
        }
    }

    // Связи между keypoints (пары индексов)
    private int[,] skeleton = new int[,]
    {
    {0,1},{0,2},{1,3},{2,4},         // голова
    {5,6},{5,7},{7,9},{6,8},{8,10},  // руки
    {5,11},{6,12},{11,12},           // торс
    {11,13},{13,15},{12,14},{14,16}  // ноги
    };

    void DrawLine(Vector2 a, Vector2 b)
    {
        if (a == Vector2.zero || b == Vector2.zero) return;
        float width = 3f;
        Vector2 diff = b - a;
        float angle = Mathf.Atan2(diff.y, diff.x) * Mathf.Rad2Deg;
        GUIUtility.RotateAroundPivot(angle, a);
        GUI.DrawTexture(new Rect(a.x, a.y - width / 2, diff.magnitude, width), lineTexture);
        GUIUtility.RotateAroundPivot(-angle, a);
    }

    void OnGUI()
    {
        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height),
                          webCamTexture, ScaleMode.ScaleToFit);
        }

        // Рисуем скелет
        if (keypoints.Count >= 17)
        {
            for (int i = 0; i < skeleton.GetLength(0); i++)
            {
                int a = skeleton[i, 0];
                int b = skeleton[i, 1];
                if (a < keypoints.Count && b < keypoints.Count)
                    DrawLine(keypoints[a], keypoints[b]);
            }

            // Рисуем точки
            foreach (var kp in keypoints)
            {
                if (kp == Vector2.zero) continue;
                GUI.DrawTexture(new Rect(kp.x - 4, kp.y - 4, 8, 8), lineTexture);
            }
        }
    }


    void OnDestroy()
    {
        worker?.Dispose();
        webCamTexture?.Stop();
    }
}