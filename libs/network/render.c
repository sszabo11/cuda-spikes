#include "config.h"
#include "data.h"
#include "encode.h"
#include "math.h"
#include "process.h"
#include "raylib.h"
#include "rlgl.h"
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const Color BG_COLOR = {1, 1, 1};

int const WIDTH = 1400;
int const HEIGHT = 950;

int const inputs = 50;
int const outputs = 50;

Vector2 *rand_cords(int n, int min_x, int max_x, int min_y, int max_y) {

  Vector2 *cords = malloc(sizeof(Vector2) * n);
  for (int i = 0; i < n; i++) {

    int rand_x = (rand() % (max_x - min_x + 1)) + min_x;
    int rand_y = (rand() % (max_y - min_y + 1)) + min_y;
    cords[i].x = rand_x;
    cords[i].y = rand_y;
  }
  return cords;
}

int render(Config *config, NetworkData *data, SpikeTrain *input_train) {
  Vector2 *cords =
      rand_cords(config->n_neurons, 200, WIDTH - 200, 50, HEIGHT - 50);
  Vector2 *cords_input = rand_cords(inputs, 50, 200, 50, HEIGHT - 50);
  Vector2 *cords_output =
      rand_cords(outputs, WIDTH - 200, WIDTH - 50, 50, HEIGHT - 50);

  InitWindow(WIDTH, HEIGHT, "Neuromesh");

  SetTargetFPS(60);
  // srand(302);

  Image image = LoadImage("../assets/bg1.png");

  Texture2D texture = LoadTextureFromImage(image);

  BeginBlendMode(BLEND_ALPHA);
  UnloadImage(image);
  printf("raylib Version: %i.%i.%i\n", RAYLIB_VERSION_MAJOR,
         RAYLIB_VERSION_MINOR, RAYLIB_VERSION_PATCH);

  Color redish = {255, 59, 0, 100};
  Color Blue1 = {0, 187, 255, 100};
  Color Blue2 = {0, 71, 153, 100};
  Color Blue3 = {0, 111, 255, 55};
  Color Blue4 = {0, 111, 255, 25};
  Color LightBlue = {175, 234, 255, 100};
  Color White2 = {255, 255, 255, 20};
  Color LightYellow = {255, 195, 0, 20};
  Color LightRed = {255, 0, 0, 20};
  Color OffWhite = {255, 255, 255, 20};
  Color OffGrey = {255, 255, 255, 1};
  Color NearWhite = {255, 255, 255, 140};

  int frame = 0;
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BG_COLOR);
    DrawTexture(texture, 0, 0, WHITE);
    DrawText("NEUROMESH", 20, 20, 16, LIGHTGRAY);

    if (frame % 3 == 0) {
      // data->pre_spikes = SPIKE(input_train, 0, 0, frame); // TImESTEOS?
      cudaError_t res = process(config, data);
    }
    for (int i = 0; i < config->n_neurons; i++) {
      // DrawCircleGradient(cords[i].x, cords[i].y, 12, Blue3, Blue4);
      // DrawCircleGradient(cords[i].x, cords[i].y, 6, Blue1, Blue2);
      if (data->post_spikes[i] == 1) {
        // DrawCircle(cords[i].x, cords[i].y, 20, WHITE);
        DrawCircleGradient((Vector2){cords[i].x, cords[i].y}, 10, WHITE,
                           White2);
        // DrawCircleGradient((Vector2){cords[i].x, cords[i].y}, 10.0, YELLOW,
        // MAROON);
        DrawCircleGradient((Vector2){cords[i].x, cords[i].y}, 4, LightBlue,
                           WHITE);
      } else {
        DrawCircleGradient((Vector2){cords[i].x, cords[i].y}, 6, NearWhite,
                           OffWhite);
        DrawCircleGradient((Vector2){cords[i].x, cords[i].y}, 4, OffWhite,
                           OffGrey);
      }
    }

    for (int i = 0; i < inputs; i++) {
      DrawCircleGradient((Vector2){cords_input[i].x, cords_input[i].y}, 10.0,
                         YELLOW, LightYellow);
      DrawCircleGradient((Vector2){cords_input[i].x, cords_input[i].y}, 6.0,
                         WHITE, YELLOW);
    }
    for (int i = 0; i < outputs; i++) {
      DrawCircleGradient((Vector2){cords_output[i].x, cords_output[i].y}, 10.0,
                         RED, LightRed);
      DrawCircleGradient((Vector2){cords_output[i].x, cords_output[i].y}, 6.0,
                         WHITE, RED);
    }
    DrawFPS(10, 10);
    EndDrawing();
    frame++;
  }

  UnloadTexture(texture);
  EndBlendMode();
  CloseWindow();
  return 0;
}
