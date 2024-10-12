import zipfile
import datetime


# Сохраняем все нужное в zip файл.
submission_zip = zipfile.ZipFile(f"submission-{datetime.datetime.now()}.zip".replace(':', '-').replace(' ', '-'), "w")
submission_zip.write('./Makefile', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('run.py', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('run.sh', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('train.py', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('train.sh', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('model.pkl', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('scripts', compress_type=zipfile.ZIP_DEFLATED)
submission_zip.write('scripts/data_utils.py', compress_type=zipfile.ZIP_DEFLATED)

submission_zip.close()