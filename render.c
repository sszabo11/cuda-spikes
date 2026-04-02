#include "config.h"
#include "data.h"
#include "math.h"
#include "raylib.h"
#include "rlgl.h"
#include <stdlib.h>
#include <time.h>

const Color BG_COLOR = {1, 1, 1};

int const WIDTH = 1400;
int const HEIGHT = 950;

int const inputs = 100;
int const outputs = 100;

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

int render(Config *config, NetworkData *data) {
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

  Color redish = {255, 59, 0};
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BG_COLOR);
    DrawTexture(texture, 0, 0, WHITE);
    DrawText("NEUROMESH", 20, 20, 16, LIGHTGRAY);

    for (int i = 0; i < config->n_neurons; i++) {
      DrawCircleGradient(cords[i].x, cords[i].y, 6, WHITE, BLUE);
    }

    for (int i = 0; i < inputs; i++) {
      DrawCircleGradient(cords_input[i].x, cords_input[i].y, 6, WHITE, YELLOW);
    }
    for (int i = 0; i < outputs; i++) {
      DrawCircleGradient(cords_output[i].x, cords_output[i].y, 6, WHITE, RED);
    }
    EndDrawing();
  }

  UnloadTexture(texture);
  EndBlendMode();
  CloseWindow();
  return 0;
}
