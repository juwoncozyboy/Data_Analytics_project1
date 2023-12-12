# train.py
# 각종 주석은 이해를 돕기 위한 것으로, 삭제하고 프로젝트를 진행하셔도 무방합니다.
# TODO가 아닌 부분도 얼마든지 삭제, 추가, 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요해질 경우 메일로 문의)

import os
import datetime
from omegaconf import OmegaConf
import logging

import torch
import dataset as module_data
import model as module_arch
import metric as module_metric
from utils import fix_seed, MetricTracker

END_DATE = '2023-10-27'  # !!! 수정 금지 !!!


def main(config, logger, tb_logger=None):

    # 시드 고정
    fix_seed(config['seed'])

    # 데이터 불러오기
    train_dataset = getattr(module_data, config.dataset.type)(end_date=END_DATE, 
                                                              is_training=True, 
                                                              **config.dataset.args)
    # 데이터로더 설정하기
    train_dataloader = getattr(torch.utils.data, config.dataloader.type)(train_dataset, 
                                                                         **config.dataloader.args)

    # TODO: 모델 선언
    input_size = train_dataset.__getitem__(0)[0].size(0)
    model = getattr(module_arch, config.model.type)(input_size=input_size, **config.model.args)

    # resume이 참일 경우, 해당 경로에서 모델 불러와서 학습 진행
    if config.train.resume:
        model.load_state_dict(torch.load(config.train.resume_path))

    # GPU 사용 설정
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #print(f'MPS 사용 가능 : {torch.backends.mps.is_available()}')

    # 모델을 GPU 메모리에 올리기
    model = model.to(device)

    # loss 및 metric 설정
    criterion = getattr(torch.nn, config.loss)().to(device)
    metrics = [getattr(module_metric, met) for met in config.metrics]
    metric_tracker_train = MetricTracker('loss', *config.metrics)
    ##추가한 매트릭스들 train에 추가하기##
    config.metrics.append('hit_rate')
    config.metrics.append('r2_score') 
    config.metrics.append('theils_u')

    # optimizer 및 scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = getattr(torch.optim, config.optimizer.type)(trainable_params, **config.optimizer.args)
    
    # L2 규제 강도를 config에서 읽어옴
    weight_decay = config.optimizer.args.get('weight_decay', 0)
    
    optimizer = getattr(torch.optim, config.optimizer.type)(
        trainable_params, 
        lr=config.optimizer.args.lr,
        weight_decay=weight_decay  # L2 규제 적용
    )
    lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.type)(optimizer, **config.lr_scheduler.args)

    # 학습 진행
    for epoch in range(1, config.train.epochs+1):
        
        # train loop
        model.train()
        metric_tracker_train.reset()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # loss 및 metric 결과 저장
            metric_tracker_train.update('loss', loss.item())
            for met in metrics:
                metric_tracker_train.update(met.__name__, met(output, target))

        # scheduler 업데이트
        lr_scheduler.step()
        
        
        # tensorboard에 loss, metric, lr 기록
        if tb_logger is not None:
            # # 방법 1) metric_name/train 에 해당하는 scalar를 기록
            # tb_logger.add_scalar('loss/train', metric_tracker_train.avg('loss'), epoch)
            # for key, value in metric_tracker_train.result().items():
            #     tb_logger.add_scalar(f'{key}/train', value, epoch)
            # tb_logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

            # 방법 2) train/metric_name 에 해당하는 scalar 기록
            for key, value in metric_tracker_train.result().items():
                tb_logger.add_scalar(f'train/{key}', value, epoch)
            tb_logger.add_scalar('train/loss', metric_tracker_train.avg('loss'), epoch)
            tb_logger.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        

        # 현재 epoch 결과를 print_period마다 출력
        print_result = (epoch % config.train.print_period) == 0
        if print_result:
            logger.info(f'[Epoch {epoch:02d}/{config.train.epochs:02d}] ' +
                        f'TRAIN ' +
                        ',  '.join([f'{k.upper()}: {v:.2f}' for k, v in metric_tracker_train.result().items()]))
            logger.info('-' * 10)
    

        # TODO: 최종 제출할 모델을 골라, teamX_model.pt로 이름을 수동 변경해주시면 됩니다.
        if epoch % config.train.save_period == 0:
            save_file_path = os.path.join(config.train.save_dir,   
                                          #f'checkpoints/{config.train.save_model_name}-e{epoch}.pt')
                                          #f'checkpoints/{config.train.save_model_name}.pt')
                                           f'checkpoints/teamF_model.pt')
                                    
            
            # 만일 이미 같은 이름의 파일이 존재한다면, 덮어쓸 것인지 물어봄
            if os.path.exists(save_file_path):
                overwrite = input(f'{save_file_path} already exists. overwrite? [y/N]')
                if overwrite != 'y':
                    logger.info(f'skip save model: {save_file_path}')
                    logger.info('-' * 10)
                    continue
            torch.save(model.state_dict(), save_file_path)
            logger.info(f'save model: {save_file_path}')  # saved/checkpoints/team_model.pt로 원상복구 해야 함
            logger.info('-' * 10)


if __name__ == "__main__":

    # 프로젝트 폴더로 이동
    src_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.dirname(src_path))  
    
    # config 불러오기
    cfg = OmegaConf.load(os.path.join('src', f'config.yaml'))

    # 각종 디렉토리 생성
    os.makedirs(os.path.join(cfg.train.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(cfg.train.save_dir, 'logs'), exist_ok=True)

    # logger 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'{cfg.train.save_dir}/logs/y{time}.log')
    logger.addHandler(file_handler)

    # tensorboard 설정
    if cfg.train.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'{cfg.train.save_dir}/tensorboard/{time}')

    # main 함수 실행
    main(config=cfg, logger=logger, tb_logger=writer if cfg.train.tensorboard else None)

    # config 출력
    logger.info('=' * 20)
    logger.info(OmegaConf.to_yaml(cfg))





