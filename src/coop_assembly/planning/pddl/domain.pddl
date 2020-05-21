(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)

    (Assembled ?e)
    (Removed ?e)

    ; (PlaceAction ?r ?e ?q1 ?q2 ?t)
    (PlaceAction ?r ?e ?t)
    (MoveAction ?r ?q2 ?t)
    (Grounded ?e)
    (Connected ?e)
    (Joined ?e1 ?e2)
    (Traj ?r ?t)

    (CollisionFree ?r ?t ?e)
    (UnSafeTraj ?r ?t)

    (CanMove ?r)
    (Conf ?r ?q)
    (AtConf ?r ?q)
    (AtStart ?q ?t)
    (Assigned ?r ?e)
    (Order ?e1 ?e2)
    ; (Stiff)
  )

;   (:action move
;     ; :parameters (?r ?q1 ?q2 ?t2)
;     :parameters (?r ?q2 ?t2)
;     :precondition (and
;                         ; (Conf ?r ?q1)
;                         ; (AtConf ?r ?q1)
;                         (Conf ?r ?q2)
;                         (Traj ?r ?t2)
;                         (CanMove ?r)
;                         (MoveAction ?r ?q2 ?t2)
;                         ;;; collision constraint
;                         (not (UnSafeTraj ?r ?t2))
;                        )
;     :effect (and
;                 ;  (not (AtConf ?r ?q1))
;                  (AtConf ?r ?q2)
;                  (not (CanMove ?r)) ; switch to avoid transit forever
;                  )
;   )

  ; place = remove the element
  (:action place
    ; :parameters (?r ?e ?q1 ?q2 ?t)
    :parameters (?r ?e ?t)
    :precondition (and
                    ;    (PlaceAction ?r ?e ?q1 ?q2 ?t)
                       (PlaceAction ?r ?e ?t)
                       (Assembled ?e)
                       ; (Stiff)
                       (Connected ?e)
                       ; e2 must be remove before e
                       (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                       ;;; Collision constraint
                    ;    (not (UnSafeTraj ?r ?t))
                       ;;; comment the following two if no transit
                    ;    (AtConf ?r ?q1) ; this will force a move action
                    ;    (not (CanMove ?r))
                       )
    :effect (and (Removed ?e)
                 (CanMove ?r)
                 (not (Assembled ?e))
                ;  (not (AtConf ?r ?q1))
                ;  (AtConf ?r ?q2)
                 )
  )

  (:derived (Connected ?e2)
   (or (Grounded ?e2)
       (exists (?e1) (and (Joined ?e1 ?e2)
                          (Assembled ?e1)
                          (Connected ?e1)
                     )
       )
   )
  )

;   (:derived (UnSafeTraj ?r ?t)
;    (exists (?e2) (and   (Element ?e2) (Traj ?r ?t) (Robot ?r)
;                         (Assembled ?e2)
;                         (not (CollisionFree ?r ?t ?e2))
;                   ))
;   )

)
