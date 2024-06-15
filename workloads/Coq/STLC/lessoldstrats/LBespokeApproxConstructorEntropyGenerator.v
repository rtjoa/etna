From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genVar' (ctx : Ctx) (t : Typ) (p : nat) (r : list nat) : list nat :=
  match ctx with
  | nil  => r
  | cons t1 ctx1 => 
    if t = t1? then
      (genVar' ctx1 t (p + 1) (cons p r))
    else
      (genVar' ctx1 t (p + 1) r)
  end.

Fixpoint genZero (env : Ctx) (tau : Typ) : G (option Expr) :=
  match tau with
  | TBool  => 
    (bindGen 
    (let _weight_true := match (tt) with
    | tt => 50
    end
    in
    freq [
      (_weight_true, returnGen true);
      (100-_weight_true, returnGen false)
    ]) 
    (fun b => 
      (returnGen (Some (Bool b)))))
  | TFun T1 T2 => 
    (bindGen (genZero (cons T1 env) T2) 
    (fun _x => match _x with
      | None  => 
        (returnGen (None ))
      | Some e => 
        (returnGen (Some (Abs T1 e)))
      end))
  end.

Fixpoint genTyp (size : nat) : G (Typ) :=
  match size with
  | O  => 
    (returnGen (TBool ))
  | S size1 => 
    (freq [
      (match (size) with
      | (1) => 55
      | (2) => 62
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (match (size) with
      | (1) => 45
      | (2) => 35
      | _ => 500
      end,
      (bindGen (genTyp size1) 
      (fun T1 => 
        (bindGen (genTyp size1) 
        (fun T2 => 
          (returnGen (TFun T1 T2)))))))])
  end.

Fixpoint genExpr (env : list Typ) (tau : Typ) (size : nat) : G (option Expr) :=
  match size with
  | O  => 
    (bindGen 
    (backtrack [
      (match (size) with
      | (0) => 60
      | _ => 500
      end,
      (oneOf_ 
      (returnGen (None )) 
      (map 
      (fun x => 
        (returnGen (Some (Var x)))) (genVar' env tau 0 (nil ))))); 
      (match (size) with
      | (0) => 39
      | _ => 500
      end,(genZero env tau))]) 
    (fun res => 
      (returnGen res)))
  | S size1 => 
    (bindGen 
    (backtrack [
      (match (size) with
      | (1) => 67
      | (2) => 69
      | (3) => 68
      | (4) => 67
      | (5) => 50
      | _ => 500
      end,
      (oneOf_ 
      (returnGen (None )) 
      (map 
      (fun x => 
        (returnGen (Some (Var x)))) (genVar' env tau 0 (nil ))))); 
      (match (size) with
      | (1) => 32
      | (2) => 29
      | (3) => 30
      | (4) => 30
      | (5) => 33
      | _ => 500
      end,
      (bindGen (genTyp 2) 
      (fun T1 => 
        (bindGen (genExpr env (TFun T1 tau) size1) 
        (fun _x => match _x with
          | None  => 
            (returnGen (None ))
          | Some e1 => 
            (bindGen (genExpr env T1 size1) 
            (fun _x => match _x with
              | None  => 
                (returnGen (None ))
              | Some e2 => 
                (returnGen (Some (App e1 e2)))
              end))
          end))))); 
      (match (size) with
      | (1) => 45
      | (2) => 42
      | (3) => 44
      | (4) => 46
      | (5) => 63
      | _ => 500
      end,match tau with
      | TBool  => 
        (bindGen 
        (let _weight_true := match (tt) with
        | tt => 50
        end
        in
        freq [
          (_weight_true, returnGen true);
          (100-_weight_true, returnGen false)
        ]) 
        (fun b => 
          (returnGen (Some (Bool b)))))
      | TFun T1 T2 => 
        (bindGen (genExpr (cons T1 env) T2 size1) 
        (fun _x => match _x with
          | None  => 
            (returnGen (None ))
          | Some e => 
            (returnGen (Some (Abs T1 e)))
          end))
      end)]) 
    (fun res => 
      (returnGen res)))
  end.

Definition gSized :=

  (bindGen (genTyp 2) 
  (fun typ => (genExpr (nil ) typ 5))).

Definition test_prop_SinglePreserve :=
forAllMaybe gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAllMaybe gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
