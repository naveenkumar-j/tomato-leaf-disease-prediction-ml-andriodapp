package com.example.plantapp;



import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;



import com.example.plantapp.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.BreakIterator;

public class MainActivity extends AppCompatActivity {


    private Button select, predict;
    private TextView textView;
    private ImageView imageView;
    private Bitmap img;
    int imageSize=224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);




        select = (Button) findViewById(R.id.select);
        predict = (Button) findViewById(R.id.predict);
        textView = (TextView) findViewById(R.id.textId);
        imageView = findViewById(R.id.imageview);



        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);

            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                img = Bitmap.createScaledBitmap(img, 384, 384, true);

                try {
                    Model model = Model.newInstance(getApplicationContext());

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 384, 384, 3}, DataType.FLOAT32);

                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();



                    float[] confidences=outputFeature0.getFloatArray();
                    int maxPos=0;
                    float maxConfidence=0;
                    for(int i=0;i<confidences.length;i++){
                        if(confidences[i]>maxConfidence){
                            maxConfidence=confidences[i];
                            maxPos=i;
                        }
                    }

                    // Releases model resources if no longer used.
                    model.close();

                    if(outputFeature0.getFloatArray()[0]>outputFeature0.getFloatArray()[1]){
                        textView.setText("Diseased leaf");
                    }
                    else{
                        textView.setText("Healthy leaf");
                    }
                } catch (IOException e) {
                    // TODO Handle the exception

                }

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 10)
        {
            imageView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
