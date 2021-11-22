import os
import uuid
from datetime import datetime

import pytz
import torch
import multiprocessing
from django_rq import job

import api.util as util
from api.image_similarity import build_image_similarity_index
from api.models.long_running_job import LongRunningJob
from api.models.photo import Photo
from api.semantic_search.semantic_search import semantic_search_instance
from api.semantic_search.server import spawnable_server


def create_batch_job(job_type, user):
    job_id = uuid.uuid4()
    lrj = LongRunningJob.objects.create(
        started_by=user,
        job_id=job_id,
        queued_at=datetime.now().replace(tzinfo=pytz.utc),
        job_type=job_type,
    )

    if job_type == LongRunningJob.JOB_CALCULATE_CLIP_EMBEDDINGS:
        batch_calculate_clip_embedding.delay(job_id, user)

    lrj.save()


@job
def batch_calculate_clip_embedding(job_id, user):
    lrj = LongRunningJob.objects.get(job_id=job_id)
    lrj.started_at = datetime.now().replace(tzinfo=pytz.utc)

    count = Photo.objects.filter(clip_embeddings__isnull=True).count()
    lrj.result = {"progress": {"current": 0, "target": count}}
    lrj.save()

    BATCH_SIZE = 64
    util.logger.info("Using threads: {}".format(torch.get_num_threads()))

    try:
        mpctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = mpctx.Pipe(duplex=True)
        server = mpctx.Process(target=spawnable_server, args=(child_conn,))
        server.start()
        done_count = 0
        while done_count < count:
            objs = list(Photo.objects.filter(clip_embeddings__isnull=True)[:BATCH_SIZE])
            valid_objs = []
            for obj in objs:
                if os.path.exists(obj.thumbnail_big.path):
                    valid_objs.append(obj)
            imgs = list(map(lambda obj: obj.thumbnail_big.path, valid_objs))
            if len(valid_objs) == 0:
                break

            parent_conn.send(imgs)
            imgs_emb, magnitudes = parent_conn.recv()
            #imgs_emb, magnitudes = semantic_search_instance.calculate_clip_embeddings(
                #imgs
            #)

            for obj, img_emb, magnitude in zip(valid_objs, imgs_emb, magnitudes):
                obj.clip_embeddings = img_emb.tolist()
                obj.clip_embeddings_magnitude = magnitude
                obj.save()
                done_count += 1

            lrj.result = {"progress": {"current": done_count, "target": count}}
            lrj.save()

        parent_conn.send(None)
        server.join()
        semantic_search_instance.unload()
        build_image_similarity_index(user)
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.finished = True
        lrj.save()

    except Exception as e:
        print(e)
