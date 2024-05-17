from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
import sys
from model_loader import load_pretaining_weights
from utils import Config, opp_malevich
import os
import numpy as np
from random import randint
from PIL import Image

TOKEN = "7011485634:AAGXQcl1ck33Z2-z1Agj-mkZf70uNbtiUCc"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class GenerationMode(StatesGroup):
    mode_1 = State()  # Режим 1: Prompt
    mode_2 = State()  # Режим 2: Prompt+Image
    mode_3 = State()  # Режим 3: Style Transfer Cyberpunk
    mode_4 = State()  # Режим 4: Style Transfer хз что



@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    keyboard = types.inline_keyboard.InlineKeyboardMarkup(
    inline_keyboard=[
        [
            types.InlineKeyboardButton(text="Режим 1: Prompt", callback_data="mode_1"),
            types.InlineKeyboardButton(text="Режим 2: Prompt+Image", callback_data="mode_2"),
            types.InlineKeyboardButton(text="Режим 3: Style Transfer Cyberpunk", callback_data="mode_3"),
            types.InlineKeyboardButton(text="Режим 4: Style Transfer хз что", callback_data="mode_4")
        ]
    ]
)

    await message.answer("Выберите режим:", reply_markup=keyboard)



@dp.callback_query_handler(lambda c: c.data and c.data in ["mode_1", "mode_2", "mode_3", "mode_4"], state='*')
async def process_mode_selection(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    selected_mode = callback_query.data
    if selected_mode == 'mode_1':
        await GenerationMode.mode_1.set()
        cfg.models = models['base']
        await bot.send_message(callback_query.from_user.id, "Введите промпт:")
    elif selected_mode == 'mode_2':
        await GenerationMode.mode_2.set()
        cfg.models = models['base']
        await bot.send_message(callback_query.from_user.id, "Введите промпт и картинку:")
    elif selected_mode == 'mode_3':
        cfg.models = models['cyber']
        await GenerationMode.mode_3.set()
        await bot.send_message(callback_query.from_user.id, "Введите промпт и картинку:")
    elif selected_mode == 'mode_4':
        cfg.models = models['anime']
        await GenerationMode.mode_4.set()
        await bot.send_message(callback_query.from_user.id, "Введите промпт и картинку:")
        


@dp.message_handler(state=GenerationMode.mode_1)
async def handle_mode_1(message: types.Message, state: FSMContext):
    print('Prompt Mode')
    name_p = randint(0, 10000)
    await bot.send_message(message.chat.id, 'Обожди')
    prompt = message.text
    try:
        image = opp_malevich(prompt, cfg)
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
    print('Done')
    # noise = np.random.normal(0, 1, (512, 512))
    
    # # Нормализуем значения шума к диапазону [0, 255]
    # noise = (noise - np.min(1)) / (np.max(1) - np.min(1)) * 255
    
    # # Преобразуем значения в целые числа
    # noise = noise.astype(np.uint8)
    
    # # Создаем изображение из массива numpy
    # image = Image.fromarray(noise)
    print(image)
    try:
        image.save(fr'C:\Users\zinov\StableDiff_scratch\data\output_{name_p}.jpg') 
        print(os.listdir(r'C:\Users\zinov\StableDiff_scratch\data'))
    except Exception as e:
        print(f"Error {e}")
    # Отправляем пользователю другое изображение
    with open(fr'C:\Users\zinov\StableDiff_scratch\data\output_{name_p}.jpg', 'rb') as new_file:
        await bot.send_photo(message.chat.id, new_file, caption='Новое изображение')
    await state.finish()  # Завершаем состояние после обработки


@dp.message_handler(state=GenerationMode.mode_2, content_types=['photo'])
async def handle_mode_2(message: types.Message, state: FSMContext):
    print('Image Mode')
    # Получаем первое изображение из списка изображений сообщения
    photo = message.photo[0]
    name_p = randint(0, 10000)
    # Получаем файл изображения
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    # Загружаем файл изображения
    file_path = await file.download(destination_file=f'image_{name_p}.jpg')
    # Получаем текст сообщения
    prompt = message.caption
    # Делаем что-то с файлом изображения и текстом сообщения
    image_gen = opp_malevich(prompt, cfg, path2image=file_path)
    image_gen.save(f'output_{name_p}.jpg') 
    # Отправляем пользователю другое изображение
    with open(f'output_{name_p}.jpg', 'rb') as new_file:
        await bot.send_photo(message.chat.id, new_file, caption='Новое изображение')
    await state.finish()  # Завершаем состояние после обработки


@dp.message_handler(state=GenerationMode.mode_3, content_types=['photo'])
async def handle_mode_3(message: types.Message, state: FSMContext):
    try:
        print('Image Mode')
        # Получаем первое изображение из списка изображений сообщения
        photo = message.photo[0]
        # Получаем файл изображения
        file_id = photo.file_id
        name_p = randint(0, 10000)
        file = await bot.get_file(file_id)
        # Загружаем файл изображения
        file_path = await file.download(destination_file=f'image_{name_p}.jpg')
        # Получаем текст сообщения
        if not message.caption:
            prompt = "student, nvinkpunk"
        else:
            prompt = message.caption + "student, nvinkpunk"
        # Делаем что-то с файлом изображения и текстом сообщения
        image_gen = opp_malevich(prompt, cfg, path2image=file_path)
        await bot.send_message(message.chat.id, 'Обожди')
        image_gen.save(f'output_{name_p}.jpg') 
        # Отправляем пользователю другое изображение
        with open(f'output_{name_p}.jpg', 'rb') as new_file:
            await bot.send_photo(message.chat.id, new_file, caption='Новое изображение')
        await state.finish()  # Завершаем состояние после обработки
    except Exception as e:
        print(f"Error: {e}")


@dp.message_handler(state=GenerationMode.mode_4, content_types=['photo'])
async def handle_mode_4(message: types.Message, state: FSMContext):
    try:
        print('Image Mode')
        # Получаем первое изображение из списка изображений сообщения
        photo = message.photo[0]
        name_p = randint(0, 10000)
        # Получаем файл изображения
        file_id = photo.file_id
        file = await bot.get_file(file_id)
        # Загружаем файл изображения
        file_path = await file.download(destination_file=f'image_{name_p}.jpg')
        file_path = os.path.abspath(file_path.name)
        # Получаем текст сообщения
        if not message.caption:
            prompt = "holliemengert artstyle"
        else:
            prompt = message.caption + "holliemengert artstyle"
        # Делаем что-то с файлом изображения и текстом сообщения
        image_gen = opp_malevich(prompt, cfg, path2image=file_path)
        image_gen.save(f'output_{name_p}.jpg') 
        # Отправляем пользователю другое изображение
        with open(f'output_{name_p}.jpg', 'rb') as new_file:
            await bot.send_photo(message.chat.id, new_file, caption='Новое изображение')
        await state.finish()  # Завершаем состояние после обработки
    except Exception as e:
        print(f'Error: {e}')



if __name__ == '__main__':
    from aiogram import executor
    models = {
        # 'base': load_pretaining_weights(r"C:\Users\zinov\Downloads\v1-5-pruned-emaonly.ckpt", 'cpu'),
        # 'cyber': load_pretaining_weights(r"C:\Users\zinov\StableDiff_scratch\data\Inkpunk-Diffusion-v2.ckpt", 'cpu'),
        'anime': load_pretaining_weights(r"C:\Users\zinov\StableDiff_scratch\data\hollie-mengert.ckpt", 'cpu')
    }
    cfg = Config()
    cfg.models = models
    print('cfg initial')
    executor.start_polling(dp, skip_updates=True)